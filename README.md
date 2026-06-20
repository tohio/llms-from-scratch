# LLMs from Scratch

A from-scratch implementation of LLM components in Python, covering both
GPT and LLaMA-style architectures. Each component is implemented as a
standalone, reusable Python module. Includes both manual implementations
and PyTorch native equivalents for direct comparison.

## Structure

| Directory | What's Inside |
|-----------|---------------|
| `tokenizer/` | Character tokenizer, custom BPE, tiktoken wrapper |
| `embeddings/` | Token embeddings, positional embeddings, RoPE |
| `attention/` | Self attention, MHA, GQA, MQA |
| `feed_forward_network/` | GELU FFN, SwiGLU FFN, MoE |
| `layer_norm/` | LayerNorm, RMSNorm |
| `residual_connections/` | Residual connection examples |
| `transformer/` | GPT-style and LLaMA-style transformer blocks |
| `minigpt/` | MiniGPT, MiniLLaMA, MiniLLaMAMoE, logit distillation, feature distillation |
| `pytorch_native/` | GPT and LLaMA rebuilt using PyTorch native components |
| `datasets/` | Custom PyTorch dataset, HuggingFace dataset, HF Hub dataset |
| `inference/` | Greedy, temperature, top-k, top-p, repetition penalty, beam search |
| `data_curation/` | Pretraining, SFT, and DPO data curation pipelines |
| `sft/` | Supervised fine tuning — pretrain then SFT |
| `dpo/` | DPO alignment — pretrain, SFT, and DPO in one file |
| `reasoning/` | Prompting techniques and full reasoning model training pipeline |
| `data/` | Training corpora and dataset files |

## Quick Start

```bash
git clone https://github.com/tohio/llms-from-scratch.git
cd llms-from-scratch
```

## Virtual Environment

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Deactivate when done:**
```bash
deactivate
```

> Add `.venv` to your `.gitignore` so it never gets committed.

## Run the Examples

**Tokenizers:**
```bash
python tokenizer/character.py
python tokenizer/bpe.py
python tokenizer/bpe_tiktoken.py
```

**Embeddings:**
```bash
python embeddings/token_embeddings.py
python embeddings/positional_embeddings.py
python embeddings/rope.py
```

**Attention:**
```bash
python attention/self_attention.py
python attention/multi_head_attention.py
python attention/group_query_attention.py
python attention/multi_query_attention.py
```

**Feed Forward Networks:**
```bash
python feed_forward_network/ffn.py
python feed_forward_network/moe.py
```

**Transformer Blocks:**
```bash
python transformer/transformer.py
python transformer/transformer_llama.py
```

**Train a model:**
```bash
python minigpt/gpt.py
python minigpt/llama.py
python minigpt/llama_moe.py
```

**Distillation:**
```bash
python minigpt/distillation_logits.py
python minigpt/distillation_features.py
```

**PyTorch Native Models:**
```bash
python pytorch_native/gpt.py
python pytorch_native/llama.py
```

**Datasets:**
```bash
python datasets/custom_dataset.py
python datasets/hf_dataset.py
python datasets/hf_hub_dataset.py
```

**Inference:**
```bash
python inference/greedy.py
python inference/temperature.py
python inference/top_k.py
python inference/top_p.py
python inference/repetition_penalty.py
python inference/beam_search.py
```

**Data Curation:**
```bash
python data_curation/pretrain_curate.py
python data_curation/sft_curate.py
python data_curation/dpo_curate.py
```

**Supervised Fine Tuning:**
```bash
python sft/sft.py
```

**DPO Alignment:**
```bash
python dpo/dpo.py
```

**Reasoning — Prompting Techniques:**
```bash
python reasoning/reasoning_prompt.py
```

**Reasoning — Full Reasoning Model:**
```bash
python reasoning/reasoning_model.py
```

## Models

### MiniGPT
A GPT-style language model trained autoregressively on a small corpus.

| Component | Implementation |
|-----------|---------------|
| Tokenizer | tiktoken (GPT-4o) |
| Positional Encoding | Learned positional embeddings |
| Attention | Masked Multi-Head Attention |
| Normalisation | LayerNorm |
| FFN | GELU |

### MiniLLaMA
A LLaMA-style language model — same training setup, modern architecture choices.

| Component | Implementation |
|-----------|---------------|
| Tokenizer | tiktoken (GPT-4o) |
| Positional Encoding | RoPE (Rotary Position Embedding) |
| Attention | Grouped Query Attention |
| Normalisation | RMSNorm |
| FFN | SwiGLU |

### MiniLLaMAMoE
A LLaMA 4-style model — replaces the dense FFN with a Mixture of Experts layer.

| Component | Implementation |
|-----------|---------------|
| Tokenizer | tiktoken (GPT-4o) |
| Positional Encoding | RoPE (Rotary Position Embedding) |
| Attention | Grouped Query Attention |
| Normalisation | RMSNorm |
| FFN | MoE (8 experts, top-2 routing) |

### PyTorch Native
Both models rebuilt using PyTorch native components for direct comparison
with the manual implementations.

| Component | GPT | LLaMA |
|-----------|-----|-------|
| Attention | `nn.TransformerEncoderLayer` | `F.scaled_dot_product_attention` + GQA |
| Normalisation | `nn.LayerNorm` | `nn.RMSNorm` |
| FFN | `nn.TransformerEncoderLayer` | SwiGLU |
| Position | Learned embeddings | RoPE |

### Distillation
Two distillation approaches — logit distillation (Hinton 2015) and feature distillation (TinyBERT/DistilBERT style).

| Approach | Script | Loss |
|----------|--------|------|
| Logit distillation | `minigpt/distillation_logits.py` | Hard CE + soft KL divergence |
| Feature distillation | `minigpt/distillation_features.py` | Hard CE + MSE on hidden states |

Both use a teacher/student setup — teacher trains first, student learns from teacher outputs.

## Model Comparison

Results on `tiny_corpus.txt` after 5000 training steps with identical hyperparameters.

| Model | Final Loss | Architecture |
|-------|-----------|--------------|
| MiniGPT | 0.5097 | MHA + LayerNorm + GELU |
| MiniLLaMA | 0.4221 | GQA + RMSNorm + SwiGLU + RoPE |
| MiniLLaMAMoE | 0.3847 | GQA + RMSNorm + MoE + RoPE |

## Training Pipeline

The full training pipeline mirrors production LLM development — data curation
through to preference alignment.

| Stage | Script | Description |
|-------|--------|-------------|
| Pretraining data | `data_curation/pretrain_curate.py` | FineWeb-Edu + Dolma → cleaned corpora |
| SFT data | `data_curation/sft_curate.py` | ultrachat + tulu → instruction pairs |
| DPO data | `data_curation/dpo_curate.py` | ultrafeedback + hh-rlhf → preference pairs |
| Pretraining | `minigpt/gpt.py` or `minigpt/llama.py` | Train from scratch on corpus |
| SFT | `sft/sft.py` | Fine tune on instruction/response pairs |
| DPO | `dpo/dpo.py` | Align using chosen/rejected preference pairs |

## Reasoning Model Pipeline

A three-stage pipeline for training a reasoning model — pretraining, SFT on
reasoning traces, and RL with GRPO.

| Stage | Script | Dataset | Description |
|-------|--------|---------|-------------|
| Pretrain | `reasoning/reasoning_model.py` | Sherlock + Darwin | General language and reasoning patterns |
| SFT | `reasoning/reasoning_model.py` | Nemotron chat | Teaches `<think>` format and controllable reasoning on/off |
| GRPO | `reasoning/reasoning_model.py` | Nemotron math | Rewards correct `\boxed{}` answers — verifiable RL signal |

## Inference Techniques

| Technique | Description |
|-----------|-------------|
| Greedy | Always picks the highest probability token — deterministic baseline |
| Temperature | Scales logits before sampling — controls randomness |
| Top-K | Samples only from the K most likely tokens |
| Top-P | Samples from the smallest set of tokens whose cumulative probability exceeds P |
| Repetition Penalty | Discourages the model from repeating previously generated tokens |
| Beam Search | Maintains N candidate sequences and returns the highest scoring one |

## Reasoning Techniques

`reasoning/reasoning_prompt.py` demonstrates four prompting techniques on a pretrained model.

| Technique | Description |
|-----------|-------------|
| Chain of Thought | Prompts the model to reason step by step before answering |
| Self Consistency | Samples multiple reasoning paths and takes a majority vote |
| ReAct | Interleaves reasoning with actions and observations |
| Tree of Thought | Explores multiple reasoning branches and keeps the most promising |

`reasoning/reasoning_model.py` trains an actual reasoning model using SFT on real CoT traces
and GRPO reinforcement learning with a verifiable reward signal.

## Data Files

| File | Generated By | Used By |
|------|-------------|---------|
| `data/tiny_corpus.txt` | — | pretraining, SFT, DPO, inference |
| `data/sft_dataset.jsonl` | — | `sft/sft.py` |
| `data/dpo_dataset.jsonl` | — | `dpo/dpo.py` |
| `data/sherlock_corpus.txt` | `reasoning/reasoning_prompt.py` | `reasoning/reasoning_prompt.py`, `reasoning/reasoning_model.py`, `minigpt/distillation_logits.py`, `minigpt/distillation_features.py` |
| `data/darwin_corpus.txt` | `reasoning/reasoning_prompt.py` | `reasoning/reasoning_prompt.py`, `reasoning/reasoning_model.py`, `minigpt/distillation_logits.py`, `minigpt/distillation_features.py` |
| `data/reasoning_corpus.txt` | `reasoning/reasoning_prompt.py` | `reasoning/reasoning_prompt.py`, `reasoning/reasoning_model.py`, `minigpt/distillation_logits.py`, `minigpt/distillation_features.py` |
| `data/fineweb_corpus.txt` | `data_curation/pretrain_curate.py` | pretraining |
| `data/dolma_corpus.txt` | `data_curation/pretrain_curate.py` | pretraining |
| `data/mixed_corpus.txt` | `data_curation/pretrain_curate.py` | pretraining |
| `data/fineweb_sft.jsonl` | `data_curation/sft_curate.py` | `sft/sft.py` |
| `data/dolma_sft.jsonl` | `data_curation/sft_curate.py` | `sft/sft.py` |
| `data/mixed_sft.jsonl` | `data_curation/sft_curate.py` | `sft/sft.py` |
| `data/fineweb_dpo.jsonl` | `data_curation/dpo_curate.py` | `dpo/dpo.py` |
| `data/dolma_dpo.jsonl` | `data_curation/dpo_curate.py` | `dpo/dpo.py` |
| `data/mixed_dpo.jsonl` | `data_curation/dpo_curate.py` | `dpo/dpo.py` |

## Requirements

- Python 3.10+
- PyTorch 2.4+ (required for `nn.RMSNorm`)
- tiktoken
- datasets (HuggingFace)
- transformers (HuggingFace)
- langdetect (data curation language filtering)

## Install Dependencies

```bash
pip install -r requirements.txt
```

If installing torch with CUDA support, use the correct command from
[PyTorch's official website](https://pytorch.org/get-started/locally/).

## Related

- [slm](https://github.com/tohio/slm) — production LLM pipeline built on the same architecture — data curation, pretraining, SFT, DPO, and serving at 125m/350m/1b scale

## Credits
Inspired by the work of [Rohit Kumar Tiwari](https://github.com/analyticalrohit/llms-from-scratch)

## License
MIT