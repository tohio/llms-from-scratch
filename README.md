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
| `feed_forward_network/` | GELU FFN, SwiGLU FFN |
| `layer_norm/` | LayerNorm, RMSNorm |
| `residual_connections/` | Residual connection examples |
| `transformer/` | GPT-style and LLaMA-style transformer blocks |
| `minigpt/` | MiniGPT and MiniLLaMA — full trainable models |
| `pytorch_native/` | GPT and LLaMA rebuilt using PyTorch native components |
| `datasets/` | Custom PyTorch dataset, HuggingFace dataset, HF Hub dataset |
| `inference/` | Greedy, temperature, top-k, top-p, repetition penalty, beam search |
| `data/` | tiny_corpus.txt — training corpus |

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

**Transformer Blocks:**
```bash
python transformer/transformer.py
python transformer/transformer_llama.py
```

**Train a model:**
```bash
python minigpt/gpt.py
python minigpt/llama.py
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

### PyTorch Native
Both models rebuilt using PyTorch native components for direct comparison
with the manual implementations.

| Component | GPT | LLaMA |
|-----------|-----|-------|
| Attention | `nn.TransformerEncoderLayer` | `F.scaled_dot_product_attention` + GQA |
| Normalisation | `nn.LayerNorm` | `nn.RMSNorm` |
| FFN | `nn.TransformerEncoderLayer` | SwiGLU |
| Position | Learned embeddings | RoPE |

## Inference Techniques

| Technique | Description |
|-----------|-------------|
| Greedy | Always picks the highest probability token — deterministic baseline |
| Temperature | Scales logits before sampling — controls randomness |
| Top-K | Samples only from the K most likely tokens |
| Top-P | Samples from the smallest set of tokens whose cumulative probability exceeds P |
| Repetition Penalty | Discourages the model from repeating previously generated tokens |
| Beam Search | Maintains N candidate sequences and returns the highest scoring one |

## Requirements

- Python 3.10+
- PyTorch 2.4+ (required for `nn.RMSNorm`)
- tiktoken
- datasets (HuggingFace)

## Install Dependencies

```bash
pip install -r requirements.txt
```

If installing torch with CUDA support, use the correct command from
[PyTorch's official website](https://pytorch.org/get-started/locally/).

## Credits
Inspired by the work of [Rohit Kumar Tiwari](https://github.com/analyticalrohit/llms-from-scratch)

## License
MIT