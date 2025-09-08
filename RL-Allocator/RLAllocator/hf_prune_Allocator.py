# hf_prune.py  — with embedded RL allocator (per-iteration layer sparsity assignment)
import os
import gc
import sys
import json
import math
import random
import argparse
from typing import List, Tuple, Dict, Any

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Common RMSNorm variants for custom pruners
try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
except Exception:
    LlamaRMSNorm = None
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
except Exception:
    Qwen2RMSNorm = None
try:
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
except Exception:
    Gemma2RMSNorm = None

# LLM-Pruner components
import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.datasets.example_samples import get_examples

# Your RL allocator (same directory as this file)
# Expected to expose RLAdaptor class with .load(ckpt) and .assign(...) methods
from rl_adaptor1 import RLAdaptor, LayerFeat


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
    os.makedirs(out_dir, exist_ok=True)

    # Save generation_config (if present)
    try:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.save_pretrained(out_dir)
    except Exception:
        pass

    # Save weights and config (prefer safetensors)
    model.save_pretrained(out_dir, safe_serialization=bool(safe_serialization))

    # Save tokenizer
    if save_tokenizer and tokenizer is not None:
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception:
            pass

    # Metadata
    meta = {"llm_pruner_note": "Exported pruned model as HF directory.",
            "safe_serialization": bool(safe_serialization)}
    try:
        with open(os.path.join(out_dir, "prune_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    try:
        logger and logger.log(f"Finish pruning and save HF model to {out_dir}")
    except Exception:
        print(f"[HF-EXPORT] saved to: {out_dir}")


def _save_hf_and_optional_bin(model, tokenizer, args, logger):
    output_pth = getattr(args, "output_pth", None)
    out_dir = os.path.dirname(output_pth) if output_pth else "."
    os.makedirs(out_dir, exist_ok=True)
    hf_dir_cli = getattr(args, "save_pretrained_dir", None)
    hf_dir = hf_dir_cli or os.path.join(out_dir, "hf_pruned")
    use_safe = bool(getattr(args, "safe_serialization", False))

    export_hf_dir(model, tokenizer, hf_dir, save_tokenizer=True,
                  safe_serialization=use_safe, logger=logger)

    if bool(getattr(args, "save_model", False)) and (not bool(getattr(args, "no_save_bin", False))) and output_pth:
        try:
            torch.save(model.state_dict(), output_pth)
            logger and logger.log(f"Also saved raw state_dict to {output_pth}")
        except Exception as e:
            logger and logger.log(f"[WARN] failed to save state_dict bin: {e}")


def update_model_config_after_compression(model):
    """
    Refresh config based on actual weights
    (per-layer num_heads/intermediate_size; global hidden_size/head_dim)
    """
    layers = model.model.layers
    num_attention_heads = []
    num_key_value_heads = []
    intermediate_size = []

    first_head_dim = layers[0].self_attn.head_dim
    for i, layer in enumerate(layers):
        assert layer.self_attn.head_dim == first_head_dim, f"Layer {i} head_dim mismatch"
        q_out = layer.self_attn.q_proj.weight.data.shape[0]
        k_out = layer.self_attn.k_proj.weight.data.shape[0]
        num_heads = q_out // layer.self_attn.head_dim
        num_kv_heads = k_out // layer.self_attn.head_dim

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


# ----------------- Proxies & Tiny Accuracy Probe -----------------

def _estimate_layer_proxies(model, seq_len: int) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Return four lists (F, B, A, K), one per layer.
    These are rough relative-scale estimates used for reward only.
    """
    layers = model.model.layers
    F, B, A, K = [], [], [], []
    d_model = model.model.embed_tokens.weight.shape[1]
    for i, layer in enumerate(layers):
        h = int(getattr(layer.self_attn, 'num_heads', 0) or 0)
        hk = int(getattr(layer.self_attn, 'num_key_value_heads', max(1, h // 8)))
        d_head = int(getattr(layer.self_attn, 'head_dim', d_model // max(1, h or 1)))
        inter = int(layer.mlp.gate_proj.out_features)

        # Approximate FLOPs (attention linear + 2-layer MLP)
        flops_attn = 4.0 * d_model * h * d_head
        flops_mlp  = 2.0 * d_model * inter
        F.append(flops_attn + flops_mlp)

        # Bytes (parameters + activations)
        bytes_params = (h*d_head*d_model + hk*d_head*d_model +  # q,k
                        h*d_head*d_model + d_model*h*d_head +   # v,o
                        d_model*inter + inter*d_model) * 2.0    # fp16
        bytes_act = (d_model + h*d_head + inter) * 2.0
        B.append(bytes_params + bytes_act)

        # Peak activations
        A.append(max(d_model * max(1, h), inter))

        # KV cache (decode)
        K.append(2.0 * hk * d_head)
    return F, B, A, K


@torch.no_grad()
def _tiny_ppl_probe(model, tokenizer, dataset_path: str, n: int, maxlen: int, device: str) -> float:
    """
    Minimal PPL evaluation:
    can point to a tokenized dataset directory (datasets.load_from_disk),
    sample n examples for an estimate. Return 0 if unavailable (to ignore).
    """
    try:
        from datasets import load_from_disk
        if os.path.isdir(dataset_path):
            ds = load_from_disk(dataset_path)
            col = 'input_ids' if 'input_ids' in ds['train'].features else None
            if col:
                it = ds['train'].shuffle(seed=1).select(range(min(n, len(ds['train']))))
                tot_loss, tot_tok = 0.0, 0
                model.eval()
                for ex in it:
                    ids = ex[col][:maxlen]
                    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                    out = model(ids, labels=ids)
                    loss = out.loss.detach().float().item()
                    tot_loss += loss * ids.numel()
                    tot_tok  += ids.numel()
                if tot_tok > 0:
                    return tot_loss / tot_tok
    except Exception:
        pass
    return 0.0


def _apply_layer_sparsity_to_pruner(pruner, s_list: List[float], logger=None) -> None:
    """
    Try to write per-layer sparsity back to the pruner:
    1) Prefer calling common setters;
    2) Otherwise write to common attribute names (dict/list forms).
    """
    num_layers = len(s_list)
    s_dict = {i: float(s_list[i]) for i in range(num_layers)}

    # 1) Common setters
    for api in ['set_layer_sparsity_by_index', 'update_layerwise_sparsity', 'set_layerwise_sparsity']:
        if hasattr(pruner, api):
            try:
                getattr(pruner, api)(s_dict)
                logger and logger.log(f"[RL-ALLOC] Applied via pruner.{api}")
                return
            except Exception as e:
                logger and logger.log(f"[RL-ALLOC] pruner.{api} failed: {e}")

    # 2) Attribute write-back
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
    logger and logger.log(f"[RL-ALLOC] Applied via attribute writeback: {sorted(set(touched))}")


# ------------------------- Main -------------------------

def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name=f"{args.save_log_name}",
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # --------- Load model & tokenizer ----------
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
        # Compatibility for legacy .bin package (a dict containing {'tokenizer','model'})
        pruned_dict = torch.load(args.base_model, map_location='cpu', weights_only=False)
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.log(f"Original model: {args.base_model}")

    for p in model.parameters():
        p.requires_grad_(True)

    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Original parameters: {before_pruning_parameters}")

    # Dummy input for dependency graph (values don't matter)
    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
    ], device=device)

    # --------- Importance ----------
    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']
    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = hf_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = hf_pruner.MagnitudeImportance(p=2)
    else:  # 'taylor'
        imp = hf_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)

    # Adaptive layerwise distribution (can be combined with RL; skip if --rl_allocator_only)
    layer_imp = None
    if args.adpative_prune and (not args.rl_allocator_only):
        method = args.layer_imp_method.lower()
        assert method in ['cosine', 'euclidean', 'manhattan']
        if method == 'cosine':
            layer_imp = hf_pruner.cosine; lower_is_better = True
        elif method == 'euclidean':
            layer_imp = hf_pruner.euclidean; lower_is_better = True
        else:
            layer_imp = hf_pruner.manhattan; lower_is_better = True
    else:
        lower_is_better = True  # placeholder

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
        customized = {}
        if LlamaRMSNorm: customized[LlamaRMSNorm] = hf_pruner.hf_rmsnorm_pruner
        if Qwen2RMSNorm: customized[Qwen2RMSNorm] = hf_pruner.hf_rmsnorm_pruner
        if Gemma2RMSNorm: customized[Gemma2RMSNorm] = hf_pruner.hf_rmsnorm_pruner
        # Fallback: detect any *RMSNorm*/Norm class and add it
        if not customized:
            for module in model.modules():
                if any(x in module.__class__.__name__.lower() for x in ['rmsnorm', 'norm']):
                    customized[module.__class__] = hf_pruner.hf_rmsnorm_pruner
        kwargs["customized_pruners"] = customized

        logger.log(f"Pruning Attention Layer = {list(range(args.block_attention_layer_start, args.block_attention_layer_end))}")
        logger.log(f"Pruning MLP Layer = {list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))}")

        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()

        logger.log("Start Pruning")

        # ------- RL allocator prep -------
        rl_enabled = bool(args.rl_allocator)
        adaptor = None
        if rl_enabled:
            adaptor = RLAdaptor.load(args.rl_allocator_ckpt) if args.rl_allocator_ckpt else RLAdaptor()

        if before_pruning_parameters > args.target_param_num:
            for step in range(1, 1 + args.iterative_steps):
                # ---- Taylor backprop (only when pruner_type='taylor') ----
                if pruner_type == 'taylor':
                    example_prompts = get_examples(args.calibration_data_path, tokenizer, args.num_examples, seq_len=args.taylor_seq_len).to(device)
                    logger.log(f"Start Backwarding in iterative steps = {step}...")
                    total_loss = []
                    for mini in torch.split(example_prompts, args.batch_size):
                        loss = model(mini, labels=mini).loss
                        total_loss.append(loss)
                        loss.backward()
                    logger.log(f"Average Loss = {sum(total_loss)/len(total_loss)}")

                # ---- Adapt-Pruner distribution (optional) ----
                if args.adpative_prune and (not args.rl_allocator_only):
                    _ = pruner.adaptive_update_prune_distribution(
                        example_prompts if pruner_type=='taylor' else
                        get_examples(args.calibration_data_path, tokenizer, args.batch_size, seq_len=args.taylor_seq_len).to(device),
                        lower_is_better,
                        args.layer_prune_distribution_amplitude,
                        args.batch_size
                    )

                # ---- RL allocator: reassign per-iteration per-layer sparsity ----
                if rl_enabled and (step % max(1, args.rl_allocator_every) == 0):
                    layers = model.model.layers
                    num_layers = len(layers)

                    # 1) Normalize proxy metrics (F, B, A, K)
                    F, B, A, K = _estimate_layer_proxies(model, seq_len=args.rl_seq_len)
                    def _norm(x):
                        xm = float(sum(x) / max(1, len(x)))
                        return [xi / max(xm, 1e-8) for xi in x]
                    Fh, Bh, Ah, Kh = _norm(F), _norm(B), _norm(A), _norm(K)

                    # 2) Optional tiny PPL probe (ΔPPL handled inside adaptor or ignored)
                    S_acc = 0.0
                    if args.rl_probe_dataset:
                        S_acc = -_tiny_ppl_probe(model, tokenizer, args.rl_probe_dataset,
                                                 args.rl_probe_samples, args.rl_probe_maxlen, device)

                    # 3) This iteration's target average sparsity (linearly derived from target_param_num)
                    if args.rl_target_avg is not None:
                        target_avg = float(args.rl_target_avg)
                    else:
                        cur_target_param_num = int(
                            before_pruning_parameters -
                            (before_pruning_parameters - args.target_param_num) * (step / max(1, args.iterative_steps))
                        )
                        target_avg = max(0.0, min(1.0, 1.0 - float(cur_target_param_num) / float(before_pruning_parameters)))

                    # 4) Randomize weights (improve robustness)
                    if args.rl_randomize_weights:
                        alpha = random.uniform(0.5 * args.rl_alpha, 1.5 * args.rl_alpha)
                        beta  = random.uniform(0.5 * args.rl_beta,  1.5 * args.rl_beta)
                        gamma = random.uniform(0.5 * args.rl_gamma, 1.5 * args.rl_gamma)
                        delta = random.uniform(0.5 * args.rl_delta, 1.5 * args.rl_delta)
                    else:
                        alpha, beta, gamma, delta = args.rl_alpha, args.rl_beta, args.rl_gamma, args.rl_delta

                    # ---------- Produce per-layer sparsity (LayerFeat style) ----------
                    layers = model.model.layers
                    num_layers = len(layers)
                    d_model = model.model.embed_tokens.weight.shape[1]

                    # Reuse the four normalized proxy vectors computed above
                    def _norm(x):
                        xm = float(sum(x) / max(1, len(x)))
                        return [xi / max(xm, 1e-8) for xi in x]
                    Fh, Bh, Ah, Kh = _norm(F), _norm(B), _norm(A), _norm(K)

                    # Build LayerFeat list; to be compatible with different dataclass fields, filter by keys
                    layer_feats = []
                    valid_keys = set(getattr(LayerFeat, "__dataclass_fields__", {}).keys())
                    for i, layer in enumerate(layers):
                        n_heads = int(getattr(layer.self_attn, 'num_heads', 0) or 0)
                        n_kv    = int(getattr(layer.self_attn, 'num_key_value_heads', max(1, n_heads // 8)))
                        head_dim = int(getattr(layer.self_attn, 'head_dim', max(1, d_model // max(1, n_heads or 1))))
                        inter    = int(layer.mlp.gate_proj.out_features)
                        depth    = float(i) / max(1, num_layers - 1)

                        # Candidate feature dict (only keep fields existing in LayerFeat)
                        cand = {
                            'width': float(d_model),
                            'n_heads': float(n_heads),
                            'n_kv_heads': float(n_kv),
                            'head_dim': float(head_dim),
                            'mlp_intermediate': float(inter),
                            'depth': depth,
                            # Also feed the four proxies (if LayerFeat lacks them, they’ll be filtered out)
                            'flops_proxy': float(Fh[i]),
                            'bytes_proxy': float(Bh[i]),
                            'act_proxy': float(Ah[i]),
                            'kv_proxy': float(Kh[i]),
                        }
                        feat = LayerFeat(**{k: v for k, v in cand.items() if k in valid_keys})
                        layer_feats.append(feat)

                    # Target average sparsity (reuse variable target_avg computed above)
                    # Reward weights: rl_adaptor defaults use keys like alpha (PPL penalty), b_latency/b_mem/b_flops, etc.
                    reward_weights = {
                        'alpha': float(args.rl_w_acc),       # accuracy term weight (-ΔPPL)
                        'b_flops': float(args.rl_alpha),     # FLOPs proxy weight
                        'b_mem':   float(args.rl_beta + args.rl_delta),  # bytes + KV combined as "mem"
                        'b_latency': float(args.rl_gamma),   # treat activations as a latency proxy (or 0 if not applicable)
                    }
                    # Note: if your rl_adaptor1.py uses different key names, align the dict keys to its implementation.

                    # Call assign (older signature: layer_feats + target_avg + optional reward_weights/S_acc)
                    try:
                        s_list = adaptor.assign(
                            layer_feats=layer_feats,
                            target_avg=target_avg,
                            reward_weights=reward_weights,
                            S_acc=S_acc  # compatible even if the adaptor ignores it
                        )
                    except TypeError:
                        # Fallback to older signature (without reward_weights or S_acc)
                        try:
                            s_list = adaptor.assign(layer_feats=layer_feats, target_avg=target_avg)
                        except TypeError:
                            # One more fallback: positional-only
                            s_list = adaptor.assign(layer_feats, target_avg)

                    assert isinstance(s_list, (list, tuple)) and len(s_list) == num_layers, \
                        "RLAdaptor.assign must return list[float] of length = num_layers"

                    # Write back to pruner (reuse helper)
                    _apply_layer_sparsity_to_pruner(pruner, list(map(float, s_list)), logger)

                # ---- Perform this pruning step ----
                pruner.step()

                if pruner_type == 'taylor':
                    model.zero_grad()

                gc.collect(); torch.cuda.empty_cache()

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
        gc.collect(); torch.cuda.empty_cache()

    elif args.layer_wise:
        # Layer removal (rarely used)
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
        raise NotImplementedError("Choose --block_wise or --layer_wise")

    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"#Param before: {before_pruning_parameters}, #Param after: {after_pruning_parameters}, Ratio = {100.0*after_pruning_parameters/before_pruning_parameters:.4f}%")

    _save_hf_and_optional_bin(model, tokenizer, args, logger)
    logger.log(f"Compressed model's Configuration: {model.config}")


# ------------------------- Entrypoint -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning (HF) with RL allocator')

    # base
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--save_log_name', type=str, default="llama_prune")
    parser.add_argument('--output_pth', type=str, default="llama_prune/model.bin")
    parser.add_argument('--pruning_ratio', type=float, default=1.0)  # Not used to control final target; placeholder for MetaPruner
    parser.add_argument('--pruner_type', type=str, default='taylor', choices=['random','l1','l2','taylor'])

    # block-wise (channel/column) pruning
    parser.add_argument('--block_wise', action='store_true')
    parser.add_argument('--block_attention_layer_start', type=int, default=0)
    parser.add_argument('--block_attention_layer_end', type=int, default=16)
    parser.add_argument('--block_mlp_layer_start', type=int, default=0)
    parser.add_argument('--block_mlp_layer_end', type=int, default=16)

    # layer-wise removal (optional)
    parser.add_argument('--layer_wise', action='store_true')
    parser.add_argument('--prune_layer_idx', type=str, default="9,10,11,12,13,17,18,21")

    # iterative
    parser.add_argument('--iterative_steps', type=int, default=50)
    parser.add_argument('--target_param_num', type=int, default=700000000)
    parser.add_argument('--grouping_strategy', type=str, default='sum')
    parser.add_argument('--calibration_data_path', type=str, default="slimpajama")
    parser.add_argument('--taylor', type=str, default='param_first')
    parser.add_argument('--num_examples', type=int, default=512)
    parser.add_argument('--taylor_seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)

    # Adapt-Pruner (can be combined with RL; skip if --rl_allocator_only)
    parser.add_argument('--adpative_prune', action='store_true')
    parser.add_argument('--layer_imp_method', type=str, default='cosine', choices=['cosine','euclidean','manhattan'])
    parser.add_argument('--layer_prune_distribution_amplitude', type=float, default=0.02)

    # general
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_pretrained_dir', type=str, default=None)
    parser.add_argument('--safe_serialization', action='store_true')
    parser.add_argument('--no_save_bin', action='store_true')

    # ---------------- RL allocator (embedded) ----------------
    parser.add_argument('--rl_allocator', action='store_true',
                        help='Enable RL allocator (per-iteration layer sparsity assignment before pruning).')
    parser.add_argument('--rl_allocator_only', action='store_true',
                        help='If set, skip AdaptPruner distribution; RL decides sparsity alone.')
    parser.add_argument('--rl_allocator_ckpt', type=str, default=None,
                        help='Optional policy checkpoint for RL adaptor.')
    parser.add_argument('--rl_allocator_every', type=int, default=1,
                        help='Re-assign every N pruning iterations (default: 1).')

    parser.add_argument('--rl_target_avg', type=float, default=None,
                        help='Override per-iter average sparsity [0,1]; if None, derived from target_param_num.')
    parser.add_argument('--rl_seq_len', type=int, default=4096,
                        help='Proxy sequence length for F/B/A/K.')
    parser.add_argument('--rl_w_acc', type=float, default=1.0)
    parser.add_argument('--rl_alpha', type=float, default=1.0)   # FLOPs
    parser.add_argument('--rl_beta',  type=float, default=0.0)   # Bytes
    parser.add_argument('--rl_gamma', type=float, default=0.0)   # Activations
    parser.add_argument('--rl_delta', type=float, default=0.0)   # KV-cache
    parser.add_argument('--rl_randomize_weights', action='store_true',
                        help='Randomize (alpha,beta,gamma,delta) per iteration for robustness.')
    parser.add_argument('--rl_probe_dataset', type=str, default=None,
                        help='Optional tiny tokenized dataset path for quick PPL probe.')
    parser.add_argument('--rl_probe_samples', type=int, default=128)
    parser.add_argument('--rl_probe_maxlen', type=int, default=512)

    args = parser.parse_args()
    main(args)
