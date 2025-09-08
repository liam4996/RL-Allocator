import argparse
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_text_from_dataset(name: str) -> str:
    """
    Supported built-in datasets:
      - wikitext2: 'wikitext', 'wikitext-2-raw-v1', split='test'
    """
    if name == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Concatenate into one long text
        return "\n\n".join(ds["text"])
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def chunk_ids(input_ids: torch.Tensor, seq_len: int):
    """
    Split a long sequence into fixed-length chunks of seq_len.
    The final shorter tail (if any) is dropped to avoid adding instability to results.
    """
    n_tokens = input_ids.size(1)
    usable = (n_tokens // seq_len) * seq_len
    if usable == 0:
        return []
    input_ids = input_ids[:, :usable]
    chunks = input_ids.view(1, -1, seq_len)  # (1, num_chunks, seq_len)
    return [chunks[:, i, :].contiguous() for i in range(chunks.size(1))]

@torch.no_grad()
def evaluate_ppl(model, tokenizer, text: str, device: str, dtype_str: str, seq_len: int = 2048, batch_size: int = 8):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    chunks = chunk_ids(input_ids, seq_len)
    if len(chunks) == 0:
        raise ValueError(f"文本太短，无法形成一个长度为 {seq_len} 的评估块。请减小 --seq_len 或换更长文本。")

    # Evaluate by batch; each sample's loss is the mean token NLL
    total_nll = 0.0
    total_tokens = 0

    # Choose dtype
    dtype = None
    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "float32":
        dtype = torch.float32

    for i in range(0, len(chunks), batch_size):
        batch = torch.cat(chunks[i:i+batch_size], dim=0)  # (B, seq_len)
        labels = batch.clone()
        # Directly pass labels so HF computes token-level cross-entropy
        if dtype is not None:
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype) if dtype in (torch.float16, torch.bfloat16) else torch.autocast(enabled=False):
                out = model(input_ids=batch, labels=labels)
        else:
            out = model(input_ids=batch, labels=labels)

        # out.loss is averaged over tokens (ignoring -100).
        # Here each sample has the same seq_len; ignoring the one-token shift difference,
        # we can use HF's average and multiply by token count to get total NLL.
        n_tokens = batch.numel()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    ppl = math.exp(total_nll / total_tokens)
    return ppl, len(chunks), total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="HF 格式模型目录")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--dataset", type=str, default="wikitext2", help="内置评估集：wikitext2；若使用 --text_file 则忽略")
    parser.add_argument("--text_file", type=str, default=None, help="自定义评估文本路径（UTF-8）")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        # Common for LLaMA family: no pad_token, fallback to eos_token
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    if args.text_file is not None:
        assert os.path.exists(args.text_file), f"Text file not found: {args.text_file}"
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        source = f"custom file: {args.text_file}"
    else:
        text = load_text_from_dataset(args.dataset)
        source = f"dataset: {args.dataset} (wikitext-2-raw-v1 test)"

    print(f"Evaluating PPL on {source}")
    ppl, n_chunks, n_tokens = evaluate_ppl(
        model, tokenizer, text, device=device, dtype_str=args.dtype,
        seq_len=args.seq_len, batch_size=args.batch_size
    )
    print(f"Num chunks: {n_chunks}, Tokens: {n_tokens}")
    print(f"Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
