import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from typing import List
from pathlib import Path
import shutil
from datetime import datetime
import wandb
import gc
import torch
import deepspeed
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import random
import numpy as np
import time
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import torch.distributed as dist

from transformers.utils import is_flash_attn_2_available
# The original code used `assert`; switch to a warning and keep running so we can fall back to SDPA if FA2 isn't installed.
if not is_flash_attn_2_available():
    print("[warn] FlashAttention-2 not available; fallback to default attention.", flush=True)

from LLMPruner.evaluator.eval import print_memory_usage, print_gpu_memory, print_model_architecture_and_parameters
from LLMPruner.evaluator.benchmark_eval import eval_tasks_performance
from LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
from LLMPruner.datasets.ppl_dataset import get_loaders
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.eval import print_memory_usage, print_gpu_memory
from LLMPruner.templates.prompts import prompts
from LLMPruner.post_training.trainer import LogitsTrainer, CustomCosineScheduler, CustomWSDScheduler
from LLMPruner.post_training.load_dataset import get_train_val_data, load_tokenized_dataset_for_training
from LLMPruner.utils.GPU_tracker import GPUMemoryTracker, DetailedMemoryProfiler

# ---------------------- Distributed-safe wrappers ----------------------
def _dist_is_initialized() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False

def _safe_rank() -> int:
    try:
        if _dist_is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0

def _safe_world_size() -> int:
    try:
        if _dist_is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1

def _safe_barrier():
    try:
        if _dist_is_initialized():
            dist.barrier()
    except Exception:
        pass

def _safe_all_reduce(t: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    try:
        if _dist_is_initialized():
            dist.all_reduce(t, op=op)
    except Exception:
        pass
    return t
# ----------------------------------------------------------------------

# os.environ["WANDB_SILENT"] = "true"

def set_random_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # ------------------ Precision selection: bf16 / fp16 ------------------
    precision = args.precision.lower()
    if precision not in ("bf16", "fp16"):
        precision = "bf16"
    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_bf16 = (precision == "bf16")
    use_fp16 = (precision == "fp16")
    # ---------------------------------------------------------------------

    # Load Pruned Model
    if "mobilellm" in args.prune_model.lower():
        _ = AutoModelForCausalLM.from_pretrained(
            'facebook/MobileLLM-1B', 
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        del _
        torch.cuda.empty_cache()
    try:
        tokenizer_kwargs = {}
        if "mobilellm" in args.prune_model.lower():
            tokenizer_kwargs["use_fast"] = False
        tokenizer = AutoTokenizer.from_pretrained(args.prune_model, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            args.prune_model,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    except Exception:
        pruned_dict = torch.load(args.prune_model, map_location='cpu', weights_only=False)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        model = model.to(torch_dtype)
    
    # Note: whether FA2 is actually enabled depends on the environment; we keep the switch here.
    model.config.use_flash_attention = True
    model.config.use_cache = False
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Parameter Number: {before_pruning_parameters}")

    if args.dataset_tokenized:
        train_data, val_data = load_tokenized_dataset_for_training(args.data_path)
    else:
        train_data, val_data = get_train_val_data(args.data_path, tokenizer, args.cutoff_len, args.val_set_size, None)
    
    total_tokens = sum(len(sample['input_ids']) for sample in train_data)
    logger.log(f"Process tokens: {total_tokens}")
    
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    num_samples = len(train_data)
    logger.log(f"Training on {num_samples} data point")
    
    num_gpus = torch.cuda.device_count()
    steps_per_epoch = (num_samples // (args.micro_batch_size * num_gpus * gradient_accumulation_steps)) + \
                      (1 if num_samples % (args.micro_batch_size * num_gpus * gradient_accumulation_steps) != 0 else 0)
    cur_training_steps = args.train_epochs * steps_per_epoch
    
    if args.use_lora:
        # Prepare for LoRA
        model = prepare_model_for_int8_training(model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad = True
    
    if args.use_distill:
        model_kwargs = {
            "low_cpu_mem_usage": True, 
            "trust_remote_code": True, 
            "torch_dtype": torch_dtype,
            "use_cache": False
        }
        teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **model_kwargs).to(args.device)
        teacher_model.eval()
        logger.log(f"Distilled from teacher model {args.teacher_model}")
        logger.log(f"Using distillation coefficient {args.alpha}")
    else:
        teacher_model = None
    
    model = model.to(args.device)
    
    trainer = LogitsTrainer(     
        model=model,
        teacher_model=teacher_model,
        alpha=args.alpha,
        train_dataset=train_data,     
        eval_dataset=val_data,     
        tokenizer=tokenizer,     
        args=transformers.TrainingArguments(        
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,             
            optim="adamw_torch",         
            lr_scheduler_type="cosine",         # Overridden by custom scheduler
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            num_train_epochs=args.train_epochs,         
            warmup_steps=int(cur_training_steps * 0.05),
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=max(5, int(cur_training_steps * 0.005)),         
            logging_first_step=True,         
            save_strategy="steps",         
            save_steps=int(max(1, cur_training_steps * 0.99)),         
            output_dir=args.output_dir,         
            save_total_limit=10,         
            remove_unused_columns=False,         
            report_to="none",         
            run_name=args.output_dir.split('/')[-1],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            dataloader_pin_memory=True, 
        ),     
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True     
        ), 
    )
    
    trainer.create_optimizer()
    scheduler_dict = {
        "cosine": CustomCosineScheduler,
        "WSD": CustomWSDScheduler
    }

    # If explicit total_* steps are provided, automatically enter "resume across stages" mode
    # (even if --resume_previous_stages is not set).
    resume_flag = args.resume_previous_stages or (
        args.total_training_steps is not None and
        args.base_training_steps is not None and
        args.total_warmup_steps is not None
    )

    if resume_flag:
        try:
            scheduler_cls = scheduler_dict[args.lr_scheduler]
            scheduler = scheduler_cls(
                optimizer=trainer.optimizer,
                total_training_steps=args.total_training_steps,
                base_training_steps=args.base_training_steps,
                total_warmup_steps=args.total_warmup_steps,
                min_lr=args.min_learning_rate
            )
        except KeyError:
            raise NotImplementedError(f"Scheduler '{args.lr_scheduler}' not implemented")
    else:
        try:
            scheduler_cls = scheduler_dict[args.lr_scheduler]
            scheduler = scheduler_cls(
                optimizer=trainer.optimizer,
                total_training_steps=cur_training_steps,
                base_training_steps=0,
                total_warmup_steps=int(cur_training_steps * 0.05),
                min_lr=args.min_learning_rate
            )
        except KeyError:
            raise NotImplementedError(f"Scheduler '{args.lr_scheduler}' not implemented")
    trainer.lr_scheduler = scheduler
    
    if args.use_lora:
        old_state_dict = model.state_dict
        # Hook to save only LoRA parameters
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
    
    trainer.train()
    # wandb.finish()
    out_bin = os.path.join(args.output_dir, "output_model.bin")
    torch.save({'model': model.eval().cpu(), 'tokenizer': tokenizer}, out_bin)
    print(f"[post_train] Exported AdaptPruner bin to: {out_bin}")

    _safe_barrier()

    # Previously calling dist.get_rank() here could crash on single-GPU; use the safe helper.
    logger.log(f"Process {_safe_rank()} process tokens (including padding): {trainer.total_tokens}")

    # Aggregate total_tokens (only meaningful for multi-GPU)
    try:
        total_tokens_t = torch.tensor(trainer.total_tokens, device=torch.device("cuda", local_rank))
    except Exception:
        total_tokens_t = torch.tensor(trainer.total_tokens)
    total_tokens_t = _safe_all_reduce(total_tokens_t, op=dist.ReduceOp.SUM)

    if _safe_rank() == 0:
        try:
            print(f"Total tokens (including padding) processed across all processes: {total_tokens_t.item()}")
        except Exception:
            pass

    # Write back the actual completed steps (prefer trainer.state.global_step)
    actual_steps = getattr(getattr(trainer, "state", None), "global_step", None)
    if actual_steps is None:
        actual_steps = int(cur_training_steps)
    with open(f"{args.return_pth}", "w") as f:
        f.write(str(int(actual_steps)))

    # Try to destroy the process group before exit (if it exists)
    try:
        if _dist_is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    
    return actual_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type & Path
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--save_log_name', type=str, default="llama_tune", help='the path for save the log')
    parser.add_argument('--data_path', type=str, default="Open-Orca/1million-gpt-4", help='data path')
    parser.add_argument('--dataset_tokenized', action='store_true', help='whether the dataset has been tokenized')
    parser.add_argument('--output_dir', type=str, default="output_dir", help='output directory')
    parser.add_argument('--return_pth', type=str, default='return_value.out', help='path to save return value')
    parser.add_argument('--device', type=str, default="cuda", help='device')

    # Training Hyperparameters
    parser.add_argument('--use_lora', action='store_true', help='whether use lora during post training')
    parser.add_argument('--use_distill', action='store_true', help='whether use knowledge distillation during post training')
    parser.add_argument('--teacher_model', type=str, default="None", help='teacher model path')
    parser.add_argument('--alpha', type=float, default=1.0, help='distillation loss coefficient')

    # Arguments for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # Arguments for training
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--train_epochs', type=float, default=1, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='largest learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--lr_scheduler', type=str, default="cosine", help='lr scheduler type')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='value for grad norm clipping')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=0, help='validation set size')

    parser.add_argument('--resume_previous_stages', action='store_true', help='whether resume from previous training stage')
    parser.add_argument('--total_training_steps', type=int, help='total training steps for all stages')
    parser.add_argument('--base_training_steps', type=int, help='total training steps for all stages')
    parser.add_argument('--total_warmup_steps', type=int, help='total warmup steps for all stages')

    # LoRA Configuration
    parser.add_argument('--lora_r', type=int, default=128, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # LLM hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    parser.add_argument('--seed', type=int, default=42, help='seed')

    # ddp
    parser.add_argument('--local_rank', type=int, default=-1)

    # New: precision control (default keeps bf16 like the original script; FP16 is recommended on A10G)
    parser.add_argument('--precision', type=str, default="bf16", choices=["bf16", "fp16"], help="mixed precision to use")

    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version

    main(args)
