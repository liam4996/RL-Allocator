# hf_prune.py
import os
import gc
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import argparse
from typing import Tuple
import json

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.datasets.example_samples import get_examples

# RL (FLOPs-reward fine-tuning)
from rltuner_flops import (
    rl_tune_flops,
    estimate_min_block_per_layer,
    estimate_flops_weight_per_layer,
)

# ------------------------- Utils -------------------------

def set_random_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def export_hf_dir(model, tokenizer, out_dir: str,
                  save_tokenizer: bool = True,
                  safe_serialization: bool = True,
                  logger=None):
    """
    Export the in-memory pruned model as a Hugging Face directory:

      out_dir/
        config.json
        model.safetensors (or pytorch_model.bin)
        generation_config.json (if present)
        tokenizer.* (if save_tokenizer=True)
        prune_meta.json (metadata)
    """
    os.makedirs(out_dir, exist_ok=True)

    # generation_config (if present)
    try:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.save_pretrained(out_dir)
    except Exception:
        pass

    # Weights + config
    model.save_pretrained(out_dir, safe_serialization=bool(safe_serialization))

    # tokenizer
    if save_tokenizer and tokenizer is not None:
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception:
            pass

    # Metadata
    meta = {
        "llm_pruner_note": "Exported pruned model as HF directory.",
        "safe_serialization": bool(safe_serialization),
    }
    try:
        with open(os.path.join(out_dir, "prune_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if logger is not None:
        try:
            logger.log(f"Finish pruning and save HF model to {out_dir}")
        except Exception:
            print(f"[HF-EXPORT] saved to: {out_dir}")


def _save_hf_and_optional_bin(model, tokenizer, args, logger):
    """
    Called at the end of main(args):
      1) Export HF directory (optionally using safetensors)
      2) (Optional) Save a .bin state_dict backup
    """
    # Base output directory
    output_pth = getattr(args, "output_pth", None)
    out_dir = os.path.dirname(output_pth) if output_pth else "."
    os.makedirs(out_dir, exist_ok=True)

    # HF directory location (prefer --save_pretrained_dir, else default to out_dir/hf_pruned)
    hf_dir_cli = getattr(args, "save_pretrained_dir", None)
    hf_dir = hf_dir_cli or os.path.join(out_dir, "hf_pruned")

    # safetensors?
    use_safe = bool(getattr(args, "safe_serialization", False))

    # Export HF directory
    export_hf_dir(
        model, tokenizer, hf_dir,
        save_tokenizer=True,
        safe_serialization=use_safe,
        logger=logger
    )

    # (Optional) save .bin backup
    if bool(getattr(args, "save_model", False)) \
       and not bool(getattr(args, "no_save_bin", False)) \
       and output_pth:
        try:
            torch.save(model.state_dict(), output_pth)
            try:
                logger.log(f"Also saved raw state_dict to {output_pth}")
            except Exception:
                print(f"[HF-EXPORT] also saved state_dict: {output_pth}")
        except Exception as e:
            try:
                logger.log(f"[WARN] failed to save state_dict bin: {e}")
            except Exception:
                print(f"[WARN] failed to save state_dict bin: {e}")

# ------------------- Compression helpers -------------------

def update_model_config_after_compression(model):
    """
    Refresh the config according to the actual pruned weights:
      - num_attention_heads (per layer)
      - num_key_value_heads (per layer)
      - intermediate_size (per layer)
      - hidden_size (global)
      - head_dim (global)
    """
    num_attention_heads = []
    num_key_value_heads = []
    intermediate_size = []

    first_head_dim = model.model.layers[0].self_attn.head_dim

    for i, layer in enumerate(model.model.layers):
        assert layer.self_attn.head_dim == first_head_dim, \
            f"Layer {i} has inconsistent head_dim"

        num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        num_kv_heads = layer.self_attn.k_proj.weight.data.shape[0] // layer.self_attn.head_dim

        assert num_heads * layer.self_attn.head_dim == layer.self_attn.q_proj.weight.data.shape[0], \
            f"Layer {i}: Invalid num_heads calculation"
        assert num_kv_heads * layer.self_attn.head_dim == layer.self_attn.k_proj.weight.data.shape[0], \
            f"Layer {i}: Invalid num_kv_heads calculation"

        layer.self_attn.num_heads = num_heads
        layer.self_attn.num_key_value_heads = num_kv_heads
        layer.self_attn.hidden_size = num_heads * layer.self_attn.head_dim
        layer.mlp.intermediate_size = layer.mlp.gate_proj.out_features

        num_attention_heads.append(num_heads)
        num_key_value_heads.append(num_kv_heads)
        intermediate_size.append(layer.mlp.intermediate_size)

    model.config.head_dim = first_head_dim
    model.config.num_attention_heads = num_attention_heads
    model.config.num_key_value_heads = num_key_value_heads
    model.config.intermediate_size = intermediate_size
    model.config.hidden_size = model.model.embed_tokens.weight.shape[1]

    return model

# ------------------------- Main -------------------------

def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name=f"{args.save_log_name}",
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # --------- Load model/tokenizer ----------
    try:
        tokenizer_kwargs = {}
        if "mobilellm" in args.base_model.lower():
            tokenizer_kwargs["use_fast"] = False
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tokenizer_kwargs)

        model_kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True, "torch_dtype": torch.float16}
        if "gemma" in args.base_model.lower():
            model_kwargs['attn_implementation'] = 'eager'
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    except Exception:
        # Compatibility for legacy .bin packaging (a dict containing {'tokenizer','model'})
        pruned_dict = torch.load(args.base_model, map_location='cpu', weights_only=False)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']

    model.to(args.device)
    logger.log(f"Original model: {args.base_model}")

    for p in model.parameters():
        p.requires_grad_(True)

    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Original parameters: {before_pruning_parameters}")

    # Dummy input for dependency graph (values don't matter)
    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
    ]).to(args.device)

    # --------- Importance ----------
    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']
    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = hf_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = hf_pruner.MagnitudeImportance(p=2)
    else:  # taylor
        imp = hf_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)

    layer_imp = None
    if args.adpative_prune:
        method = args.layer_imp_method.lower()
        assert method in ['cosine', 'euclidean', 'manhattan']
        if method == 'cosine':
            layer_imp = hf_pruner.cosine
            lower_is_better = True
        elif method == 'euclidean':
            layer_imp = hf_pruner.euclidean
            lower_is_better = True
        else:
            layer_imp = hf_pruner.manhattan
            lower_is_better = True

    logger.log(f"Use {pruner_type} pruner...")

    # --------- Build pruner ----------
    if args.block_wise:
        kwargs = {
            "importance": imp,
            "layer_importance": layer_imp,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "consecutive_groups": {layer.self_attn.k_proj: layer.self_attn.head_dim for layer in model.model.layers},
            "root_instances": [model.model.layers[i].self_attn.k_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        # Customized RMSNorm pruner (common models)
        if 'llama' in args.base_model.lower():
            kwargs["customized_pruners"] = {LlamaRMSNorm: hf_pruner.hf_rmsnorm_pruner}
        elif 'qwen' in args.base_model.lower():
            kwargs["customized_pruners"] = {Qwen2RMSNorm: hf_pruner.hf_rmsnorm_pruner}
        elif 'gemma' in args.base_model.lower():
            kwargs["customized_pruners"] = {Gemma2RMSNorm: hf_pruner.hf_rmsnorm_pruner}
        else:
            customized = {}
            for module in model.modules():
                if any(x in module.__class__.__name__.lower() for x in ['rmsnorm', 'norm']):
                    customized[module.__class__] = hf_pruner.hf_rmsnorm_pruner
            kwargs["customized_pruners"] = customized

        logger.log(f"Pruning Attention Layer = {list(range(args.block_attention_layer_start, args.block_attention_layer_end))}")
        logger.log(f"Pruning MLP Layer = {list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))}")

        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()

        logger.log("Start Pruning")

        if before_pruning_parameters > args.target_param_num:
            for step in range(1, 1 + args.iterative_steps):
                # ---- Taylor backprop (if using Taylor) ----
                example_prompts = get_examples(args.calibration_data_path, tokenizer, args.num_examples, seq_len=args.taylor_seq_len).to(args.device)
                if pruner_type == 'taylor':
                    logger.log(f"Start Backwarding in iterative steps = {step}...")
                    total_loss = []
                    for mini in torch.split(example_prompts, args.batch_size):
                        loss = model(mini, labels=mini).loss
                        total_loss.append(loss)
                        loss.backward()
                    logger.log(f"Average Loss = {sum(total_loss)/len(total_loss)}")

                # ---- Adaptive layerwise distribution (Adapt-Pruner) ----
                if args.adpative_prune:
                    layer_imp_dict_by_index = pruner.adaptive_update_prune_distribution(
                        example_prompts, lower_is_better, args.layer_prune_distribution_amplitude, args.batch_size
                    )

                # ---- RL: FLOPs-aware per-layer micro-adjustment (before pruner.step()) ----
                if args.rl_flops_tune:
                    try:
                        num_layers = len(model.model.layers)

                        # 1) Initial per-layer sparsity
                        s_init = [0.0] * num_layers
                        fetched = False
                        for key in ['layer_sparsity', '_layer_sparsity', 'sparsity_by_layer', 'pruning_ratio_dict_by_index']:
                            if hasattr(pruner, key):
                                val = getattr(pruner, key)
                                if isinstance(val, dict) and len(val) > 0:
                                    for k, v in val.items():
                                        k = int(k)
                                        if 0 <= k < num_layers:
                                            s_init[k] = float(v)
                                    fetched = True
                                    break
                                elif isinstance(val, (list, tuple)) and len(val) == num_layers:
                                    s_init = [float(x) for x in val]
                                    fetched = True
                                    break
                        if not fetched:
                            attn_rng = set(range(args.block_attention_layer_start, args.block_attention_layer_end))
                            mlp_rng  = set(range(args.block_mlp_layer_start, args.block_mlp_layer_end))
                            pruned_layers = attn_rng.union(mlp_rng)
                            s_init = [args.pruning_ratio if i in pruned_layers else 0.0 for i in range(num_layers)]

                        # 2) FLOPs weights & minimum granularity
                        w = estimate_flops_weight_per_layer(model, context_len=args.rl_context_len)
                        min_block = estimate_min_block_per_layer(model, group_size=args.rl_group_size)

                        # 3) Bounds (±rl_max_layer_delta per layer)
                        lower = [max(0.0, s - args.rl_max_layer_delta) for s in s_init]
                        upper = [min(1.0, s + args.rl_max_layer_delta) for s in s_init]

                        # 4) Importance and depth (optional)
                        I = [0.0] * num_layers
                        if isinstance(locals().get('layer_imp_dict_by_index', None), dict):
                            tmp = [0.0] * num_layers
                            for k, v in layer_imp_dict_by_index.items():
                                k = int(k)
                                if 0 <= k < num_layers:
                                    tmp[k] = float(v)
                            m = max(tmp) or 1.0
                            I = [x / m for x in tmp]
                        depth = [i / max(1, num_layers - 1) for i in range(num_layers)]

                        # 5) External validation hook placeholder (disabled by default)
                        eval_hook = None
                        if getattr(args, 'rl_validate_every', 0):
                            def eval_hook(_s_list):  # return a metric (e.g., ppl)
                                return 0.0

                        # 6) Run RL
                        s_rl = rl_tune_flops(
                            s_init=s_init, w=w, I=I, depth=depth, min_block=min_block,
                            lower=lower, upper=upper,
                            steps=args.rl_steps, moves_per_step=args.rl_moves_per_step,
                            lr=args.rl_lr, device=args.device,
                            eval_hook=eval_hook,
                            gate_cfg={"metric": "ppl", "max_increase": args.rl_gate_ppl_increase,
                                      "validate_every": int(args.rl_validate_every or 0),
                                      "rollback": True}
                        )

                        # 7) Apply results: try setter first; otherwise attribute write-back (no wrapper)
                        applied = False
                        s_dict = {i: float(s_rl[i]) for i in range(num_layers)}
                        for api in ['set_layer_sparsity_by_index', 'update_layerwise_sparsity', 'set_layerwise_sparsity']:
                            if hasattr(pruner, api):
                                try:
                                    getattr(pruner, api)(s_dict)
                                    logger.log(f"[RL] Applied per-layer sparsity via pruner.{api} (reward=FLOPs).")
                                    applied = True
                                    break
                                except Exception as e:
                                    logger.log(f"[RL] pruner.{api} failed: {e}")

                        if not applied:
                            touched = []
                            dict_like = [
                                'pruning_ratio_dict_by_index', 'prune_ratio_dict_by_index',
                                'layer_pruning_ratio_dict', 'layerwise_pruning_ratio_dict',
                                'sparsity_dict_by_index', 'sparsity_dict',
                            ]
                            list_like = [
                                'layer_sparsity', '_layer_sparsity', 'sparsity_by_layer',
                                'per_layer_sparsity', 'layerwise_sparsity',
                                'pruning_ratio_by_layer', 'layer_pruning_ratio',
                            ]
                            for name in dict_like:
                                try:
                                    setattr(pruner, name, dict(s_dict)); touched.append(name)
                                except Exception:
                                    pass
                            vec = [s_dict.get(i, 0.0) for i in range(num_layers)]
                            for name in list_like:
                                try:
                                    setattr(pruner, name, list(vec)); touched.append(name)
                                except Exception:
                                    pass
                            if touched:
                                logger.log(f"[RL] Applied per-layer sparsity via aggressive writeback: {sorted(set(touched))}")
                            else:
                                logger.log("[RL] Aggressive writeback found no attach point; proceeding anyway.")

                    except Exception as e:
                        logger.log(f"[RL] FLOPs-aware tuning encountered an error and was skipped: {e}")

                # ---- Perform this pruning step ----
                pruner.step()

                if pruner_type == 'taylor':
                    model.zero_grad()

                gc.collect()
                torch.cuda.empty_cache()

                after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.log(f"After Iter {step}/{args.iterative_steps}, #parameters: {after_pruning_parameters}")

                model = update_model_config_after_compression(model)

                if after_pruning_parameters < args.target_param_num:
                    break

                pruner.rebuild_DG(model)

        # Cleanup
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None
        del pruner
        gc.collect()
        torch.cuda.empty_cache()

    elif args.layer_wise:
        # Layer removal (rare)
        indices_to_remove = set(int(x) for x in args.prune_layer_idx.split(','))
        new_layers = torch.nn.ModuleList([
            layer for idx, layer in enumerate(model.model.layers)
            if idx not in indices_to_remove
        ])
        model.model.layers = new_layers
        model.config.num_hidden_layers = len(new_layers)
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, 'layer_idx'):
                layer.layer_idx = idx
            if hasattr(layer.self_attn, 'layer_idx'):
                layer.self_attn.layer_idx = idx
    else:
        raise NotImplementedError

    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"#Param before: {before_pruning_parameters}, #Param after: {after_pruning_parameters}, Ratio = {100.0*after_pruning_parameters/before_pruning_parameters:.4f}%")

    _save_hf_and_optional_bin(model, tokenizer, args, logger)
    logger.log(f"Compressed model's Configuration: {model.config}")


# ------------------------- Entrypoint -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # base
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='base model name')
    parser.add_argument('--save_log_name', type=str, default="llama_prune", help='log dir name')
    parser.add_argument('--output_pth', type=str, default="llama_prune/model.bin", help='legacy .bin backup path')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='global pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type: random|l1|l2|taylor')

    # block-wise (column/channel) pruning
    parser.add_argument('--block_wise', action='store_true')
    parser.add_argument('--block_attention_layer_start', type=int, default=3)
    parser.add_argument('--block_attention_layer_end', type=int, default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, default=31)

    # layer-wise removal (optional)
    parser.add_argument('--layer_wise', action='store_true')
    parser.add_argument('--prune_layer_idx', type=str, default="9,10,11,12,13,17,18,21")

    # iterative
    parser.add_argument('--iterative_steps', type=int, default=1)
    parser.add_argument('--target_param_num', type=int, default=1)
    parser.add_argument('--grouping_strategy', type=str, default='sum')
    parser.add_argument('--calibration_data_path', type=str, default="openhermes")
    parser.add_argument('--taylor', type=str, default='param_first')
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--taylor_seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)

    # adaptive (Adapt-Pruner)
    parser.add_argument('--adpative_prune', action='store_true')
    parser.add_argument('--layer_imp_method', type=str, default='cosine')
    parser.add_argument('--layer_prune_distribution_amplitude', type=float, default=0.03)

    # general
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_pretrained_dir', type=str, default=None)
    parser.add_argument('--safe_serialization', action='store_true')
    parser.add_argument('--no_save_bin', action='store_true')

    # RL FLOPs-aware
    parser.add_argument('--rl_flops_tune', action='store_true', help='enable RL micro-adjustment (reward=FLOPs)')
    parser.add_argument('--rl_steps', type=int, default=400)
    parser.add_argument('--rl_moves_per_step', type=int, default=8)
    parser.add_argument('--rl_lr', type=float, default=1e-2)
    parser.add_argument('--rl_context_len', type=int, default=4096)
    parser.add_argument('--rl_group_size', type=int, default=64)
    parser.add_argument('--rl_max_layer_delta', type=float, default=0.01)
    parser.add_argument('--rl_validate_every', type=int, default=0)
    parser.add_argument('--rl_gate_ppl_increase', type=float, default=0.5)

    args = parser.parse_args()
    main(args)
