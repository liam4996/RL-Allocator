# coding: utf-8
import os
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Old-pickle compatibility: let torch.load find missing/new class names ----
try:
    from transformers.models.llama import modeling_llama as _llm
    if not hasattr(_llm, "LlamaSdpaAttention") and hasattr(_llm, "LlamaAttention"):
        _llm.LlamaSdpaAttention = _llm.LlamaAttention  # fallback for deserialization only
        print("[compat] Aliased LlamaSdpaAttention -> LlamaAttention for unpickling")
except Exception:
    pass

try:
    import safetensors.torch as st
except Exception:
    st = None

# -------------------- Local logger (replacing dependency on LLMPruner's built-in logger) --------------------
import logging
def _build_logger(name="eval"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s :      %(message)s"))
        logger.addHandler(h)
    return logger
eval_logger = _build_logger("eval")

# Still reuse the project's evaluation entry
from LLMPruner.evaluator.benchmark_eval import eval_tasks_performance


# -------------------- Small utilities --------------------
def str_dtype_to_torch(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if not dtype_str:
        return None
    d = dtype_str.lower()
    if d in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if d in ["fp16", "float16", "half"]:
        return torch.float16
    if d in ["fp32", "float32"]:
        return torch.float32
    return None


def _load_all_safetensors(dir_path: str) -> Dict[str, torch.Tensor]:
    files = sorted(glob.glob(os.path.join(dir_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No *.safetensors found under {dir_path}")
    if st is None:
        raise RuntimeError("safetensors is not installed but *.safetensors files exist.")
    sd: Dict[str, torch.Tensor] = {}
    for f in files:
        sd.update(st.load_file(f, device="cpu"))
    return sd


def _load_pytorch_bin(dir_path: str) -> Dict[str, torch.Tensor]:
    # Support pytorch_model.bin or sharded pytorch_model-00001-of-0000N.bin
    bin_files = sorted(glob.glob(os.path.join(dir_path, "pytorch_model*.bin")))
    if not bin_files:
        raise FileNotFoundError(f"No pytorch_model*.bin found under {dir_path}")
    sd: Dict[str, torch.Tensor] = {}
    for f in bin_files:
        part = torch.load(f, map_location="cpu")
        sd.update(part)
    return sd


def _load_state_dict_from_dir(model_dir: str) -> Dict[str, torch.Tensor]:
    st_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if st_files:
        return _load_all_safetensors(model_dir)
    return _load_pytorch_bin(model_dir)


def _overlay_state_dict(base_sd: Dict[str, torch.Tensor],
                        pruned_sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Overlay weights from pruned_sd onto base_sd (tolerate mismatched 'model.' prefixes)."""
    equal = prefix = skipped = 0
    new_sd = dict(base_sd)  # copy
    for k, v in pruned_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v
            equal += 1
            continue
        # Try adding/removing the "model." prefix
        k2 = k[6:] if k.startswith("model.") else f"model.{k}"
        if k2 in new_sd and new_sd[k2].shape == v.shape:
            new_sd[k2] = v
            prefix += 1
        else:
            skipped += 1
    stats = {
        "equal": equal,
        "prefix": prefix,
        "skipped": skipped,
        "base_total": len(base_sd),
        "pruned_total": len(pruned_sd),
    }
    print(f"[overlay] equal={equal}, prefix={prefix}, skipped={skipped}, "
          f"base_total={stats['base_total']}, pruned_total={stats['pruned_total']}")
    return new_sd, stats


def _is_varwidth_config(cfg: dict) -> bool:
    """Detect whether this is a 'per-layer variable-width' pruned config (num_attention_heads / intermediate_size etc. are lists)."""
    for key in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if key in cfg and isinstance(cfg[key], list):
            return True
    return False


# --------- Task name resolver: align social_i_qa / social_iqa / siqa to the currently available task names ----------
def resolve_tasks(raw_tasks: List[str]) -> List[str]:
    try:
        import lm_eval
        available = set(lm_eval.list_tasks())
    except Exception:
        return raw_tasks

    alias = {
        "social_i_qa": ["social_i_qa", "social_iqa", "siqa", "socialiqa"],
        "social_iqa":  ["social_iqa",  "social_i_qa", "siqa", "socialiqa"],
        "siqa":        ["siqa",        "social_iqa",  "social_i_qa", "socialiqa"],
        # Also include some common aliases
        "arc_easy":        ["arc_easy", "ai2_arc_easy", "arc_e", "arc-easy"],
        "arc_challenge":   ["arc_challenge", "ai2_arc_challenge", "arc_c", "arc-challenge"],
        "winogrande":      ["winogrande", "winogrande_xl", "winogrande-xl"],
        "hellaswag":       ["hellaswag"],
        "openbookqa":      ["openbookqa", "obqa"],
        "piqa":            ["piqa"],
    }

    def _normalize(s: str) -> str:
        return s.replace("-", "").replace("_", "").lower()

    out = []
    for t in raw_tasks:
        if t in available:
            out.append(t)
            continue
        # 1) Try alias list
        for cand in alias.get(t, []):
            if cand in available:
                out.append(cand)
                break
        else:
            # 2) Fuzzy match by normalized strings
            tn = _normalize(t)
            hit = None
            for a in available:
                if tn in _normalize(a):
                    hit = a
                    break
            out.append(hit or t)
    if raw_tasks != out:
        print(f"[tasks] resolved {raw_tasks} -> {out}")
    return out


# -------------------- Load model/tokenizer (supports var-width overlay) --------------------
def load_model_and_tokenizer(path: str,
                             base_model: Optional[str],
                             dtype_str: Optional[str],
                             device: str = "cuda"):
    """
    - If `path` is an HF directory and its config uses per-layer lists (var-width), then:
        1) Initialize the model from `base_model` (standard config)
        2) Read pruned weights from `path` and overlay them
    - Otherwise, directly use from_pretrained(path)
    - If it's a .bin, instantiate from `base_model` then overlay
    """
    torch_dtype = str_dtype_to_torch(dtype_str)

    if os.path.isdir(path):
        cfg_path = os.path.join(path, "config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path, "r"))
            # Prefer tokenizer from pruned dir; if missing, fall back to base_model
            tok_from = path if os.path.exists(os.path.join(path, "tokenizer.json")) or \
                                os.path.exists(os.path.join(path, "tokenizer.model")) else base_model or path
            tokenizer = AutoTokenizer.from_pretrained(tok_from, use_fast=True)

            if _is_varwidth_config(cfg):
                if not base_model:
                    raise ValueError(
                        "Detected var-width (per-layer) pruned config, but --base_model is not provided."
                    )
                print(f"[loader] Detected per-layer (varwidth) config. "
                      f"Overlaying pruned weights on base: {base_model}")
                # 1) base init
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch_dtype,
                    device_map=None if device == "cpu" else "auto",
                )
                model.eval()

                # 2) read pruned weights
                if glob.glob(os.path.join(path, "*.safetensors")):
                    pruned_sd = _load_all_safetensors(path)
                elif glob.glob(os.path.join(path, "pytorch_model*.bin")):
                    pruned_sd = _load_pytorch_bin(path)
                else:
                    raise FileNotFoundError(f"No model weight files found under {path}")

                # 3) overlay
                base_sd = model.state_dict()
                new_sd, _stats = _overlay_state_dict(base_sd, pruned_sd)
                missing, unexpected = model.load_state_dict(new_sd, strict=False)
                print(f"[overlay] missing={len(missing)}, unexpected={len(unexpected)}")

                if device != "cpu":
                    model.to(device)
                return model, tokenizer

            # Regular HF directory: load with from_pretrained
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                device_map=None if device == "cpu" else "auto",
            )
            model.eval()
            if device != "cpu":
                model.to(device)
            return model, tokenizer

    # If it's a .bin (raw state_dict) file
    if path.endswith(".bin"):
        if not base_model:
            raise ValueError("Loading raw .bin requires --base_model to instantiate the architecture.")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=None if device == "cpu" else "auto",
        )
        # For extracting weights only: tolerate missing legacy transformers class names
        try:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            sd = torch.load(path, map_location="cpu")
        base_sd = model.state_dict()
        new_sd, _stats = _overlay_state_dict(base_sd, sd)
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"[overlay] missing={len(missing)}, unexpected={len(unexpected)}")
        if device != "cpu":
            model.to(device)
        model.eval()
        return model, tokenizer

    # Fallback: treat `path` as an HF name/directory
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        device_map=None if device == "cpu" else "auto",
    )
    if device != "cpu":
        model.to(device)
    model.eval()
    return model, tokenizer


# -------------------- (Optional) predownload datasets: tolerant to name changes --------------------
def predownload(tasks: List[str]):
    print("[predownload]", ", ".join(tasks), "...")


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_paths", type=str, required=True,
                   help="逗号分隔：HF 目录或 .bin 文件；会逐个评测")
    p.add_argument("--base_model", type=str, default=None,
                   help="当传入的是 .bin 或 var-width HF 目录时，用它来实例化基模架构")
    p.add_argument("--eval_device", type=str, default="cuda",
                   choices=["cpu", "cuda"])
    p.add_argument("--dtype", type=str, default=None,
                   help="bfloat16 / float16 / float32")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--tasks", type=str, default=None,
                   help="逗号分隔任务，若不填用默认集")
    p.add_argument("--save_log_name", type=str, default="eval_run")
    p.add_argument("--output_dir", type=str, default="prune_log/eval_out")
    return p.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    raw_list = [s.strip() for s in args.model_paths.split(",") if s.strip()]
    print(raw_list)
    for model_path in raw_list:
        eval_logger.info(f"Evaluating model: {model_path}")
        print(f"Evaluating model: {model_path}")

        # Task resolution (with alias tolerance)
        if args.tasks:
            want = [t.strip() for t in args.tasks.split(",") if t.strip()]
        else:
            want = ["arc_easy", "arc_challenge", "hellaswag", "openbookqa", "piqa", "winogrande", "social_i_qa"]
        tasks = resolve_tasks(want)
        predownload(tasks)

        # Load model & tokenizer (supports var-width overlay)
        model, tokenizer = load_model_and_tokenizer(
            model_path, args.base_model, args.dtype, device=args.eval_device
        )

        # Evaluate
        result_table, avg_score = eval_tasks_performance(
            model, tokenizer, tasks=tasks, num_fewshot=0
        )

        # Print & save
        print("\n=== Zero-shot ===\n" + result_table)
        print(f"\nAverage score: {avg_score}\n")

        # Also evaluate wikitext2 ppl
        result_table_wiki, _ = eval_tasks_performance(
            model, tokenizer, tasks=["wikitext"], num_fewshot=0
        )
        print("\n=== WikiText2 PPL ===\n" + result_table_wiki)

        subdir = os.path.join(args.output_dir, args.save_log_name)
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "result.txt"), "w", encoding="utf-8") as f:
            f.write("=== Zero-shot ===\n")
            f.write(result_table + "\n")
            f.write(f"\nAverage score: {avg_score}\n\n")
            f.write("=== WikiText2 PPL ===\n")
            f.write(result_table_wiki + "\n")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parse_args()
    main(args)
