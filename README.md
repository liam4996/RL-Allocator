# RL-Allocator&RL-Refine

RL-Allocator is an RL-driven structured width-pruning allocator that replaces hand-crafted layer budgets with a policy optimizing a mixed reward over accuracy and deployment proxies (FLOPs/bytes/activations/KV)in this layer once forward, yielding hardware-aligned sparsity layouts under a fixed global budget.


Peak Activations are the size of the "coexisting maximum activation tensor" (peak memory usage),KV-cache is the size of the Key/Value cache that needs to be saved during inference (the autoregressive model increases each time a new token is generated) 

RL-Refine is a sum-preserving RL micro-adjustment that starts from Adapt-Pruner’s allocation and makes small, discrete transfers of heads/FFN blocks between layers, maintaining accuracy while nudging sparsity toward proxy-critical layers.

## Features

- Efficient layer-wise adaptive pruning
- A novel acceleration paradigm called Adapt-Accel, which is the first method that interleaves the pruning with training in a highly frequent manner
- Easy to use Python APIs

## Installation

```bash
git clone https://github.com/liam4996/RL-Allocator -b beta
```

Install dependent packages

```bash
pip install -r requirement.txt
```

**❗Then please replace the corresponding files of Transformers package using the code in custom_transformer_package**


## Reproduction Results
We provide scripts to prune:

1. MobileLLM-350M → 125M
2. MobileLLM-600M → 350M
3. MobileLLM-1B → 600M
4. Qwen-2.5-0.5B → 350M
5. Deepseek-R1-Distill-Qwen-1.5B → 1B   
6. Llama-3.2-1.2B → 1B


To run MobileLLM-350M → 125M experiment, the first step is to run processing dataset as:
```bash
bash process_dataset.sh bash process_datasets.sh MobileLLM-300MB PROCESS_DATA_DIR
```
And then prune the model as:
```bash
bash iterative_Prune_Train_Qwen0.5B.sh MODEL_DIR DATA_DIR
```
If you want to use RL-Allocator,you can replace hf_prune.py as hf_prune_Allocator.py in sh. Similarly, when using RL-Refine, use hf_prune_Refine.py instead.
If you just want prune，RL-Allocator：
python hf_prune_Allocator.py \
  --base_model meta-llama/Llama-3.2-1B \
  --block_wise \
  --block_attention_layer_start 0 --block_attention_layer_end 16 \
  --block_mlp_layer_start 0       --block_mlp_layer_end 16 \
  --pruning_ratio 0.20 \
  --iterative_steps 1 \
  --taylor_seq_len 64 --num_examples 512 --batch_size 16 \
  --calibration_data_path slimpajama \
  --rl_allocator \
  --rl_seq_len 4096 \
  --rl_w_acc 1.0 --rl_alpha 0.2 --rl_beta 0.6 --rl_gamma 0.2 --rl_delta 0.4 \
  --save_pretrained_dir outputs/alloc_20_pruneonly \
  --safe_serialization


RL-Refine：
python hf_prune_Refine.py \
  --base_model meta-llama/Llama-3.2-1B \
  --block_wise \
  --block_attention_layer_start 0 --block_attention_layer_end 16 \
  --block_mlp_layer_start 0       --block_mlp_layer_end 16 \
  --pruning_ratio 0.20 \
  --iterative_steps 1 \
  --taylor_seq_len 64 --num_examples 512 --batch_size 16 \
  --calibration_data_path openhermes \
  --rl_refine \
  --rl_seq_len 4096 \
  --save_pretrained_dir outputs/refine_20_pruneonly \
  --safe_serialization



change MODEL_DIR to where you want to save the model, and DATA_DIR to the directory of processed data. And the final output model will be saved in output_model folder. The model in output_model folder can be used for the evaluation.
Similar scripts for pruning other models are provided in the [experiments](experiments/) folder.

To do few-shots evaluation using TruthfulQA, AGIEval, and MMLU, run the evaluation script as:
```bash
bash eval_fewshots.sh MODEL_DIR EVAL_LOG_DIR 
```

To do zero-shot evaluation and Wikitext2 Perplexity, run the evaluation script as:
```bash
bash eval_zeroshot.sh MODEL_DIR EVAL_LOG_DIR 
```

Notice, MODEL_DIR should be the output_model directory in previously saved model files.

## Evaluation results
Following the above evaluation steps, the MMLU results are
|Model Name|MMLU|
|----|----|
|MobileLLM-350M → 125M|25.35|
|MobileLLM-600M → 350M|32.25|
|MobileLLM-1B → 600M|37.34|

## Examples

For detailed usage examples and tutorials, please refer to the [experiments](experiments/) in the repository.

## License

Apache-2.0 license

## Acknowledgments

This research used the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta is a joint effort of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.

This research used both the DeltaAI advanced computing and data resource, which is supported by the National Science Foundation (award OAC 2320345) and the State of Illinois, and the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois.. Delta and DeltaAI are joint efforts of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.

This project is built on top of [LLM-Pruner](https://github.com/horseee/LLM-Pruner), an efficient pruning library for LLM. We thank the LLM-Pruner team for providing this foundation.


