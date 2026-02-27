# KV-CLOAK

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Official implementation of **"Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference"** (NDSS 2026)

[Paper](https://www.ndss-symposium.org/ndss-paper/shadow-in-the-cache-unveiling-and-mitigating-privacy-risks-of-kv-cache-in-llm-inference/) • [Project Page](#) • [Citation](#citation)

---

## Overview

KV-CLOAK is a privacy-preserving protection mechanism for Key-Value (KV) caches in Large Language Model (LLM) inference. It defends against three types of privacy attacks that can reconstruct user inputs from cached intermediate states:

- **Inversion Attack**: Reconstructs input tokens from KV-cache
- **Collision Attack**: Uses distance metrics to recover tokens
- **Injection Attack**: Injects malicious prompts to extract information

### Key Results

| Attack Type | Metric | Origin | KV-Cloak | Reduction |
|-------------|--------|--------|----------|-----------|
| **Inversion** | BERTScore | 1.000 | 0.050 | **95%** ↓ |
| **Collision+** | BERTScore | 1.000 | 0.077 | **92%** ↓ |
| **Injection** | ROUGE-L | 0.302 | 0.000 | **100%** ↓ |

*Tested on Llama-3.2-1B with 20 samples*

### Performance Overhead

| Method | Latency (64x256) | Latency (64x512) |
|--------|-----------------|------------------|
| Origin | baseline | baseline |
| AES | 3,378 ms | 7,889 ms |
| **KV-Cloak (fused)** | **34 ms** | **57 ms** |
| DP | 12 ms | 23 ms |

*KV-Cloak is ~100-200× faster than AES encryption while providing comparable protection*

### Architecture

```
User Input → [Model] → KV-Cache → [KV-Cloak] → Protected Cache
                ↓                        ↓
          (Inference)            (Encryption)
                ↓                        ↓
          Attack Attempt ← [Defense] ← Decryption
```

---

## Quick Start

### 1. Setup Environment

```bash
conda create --name kvcloak python=3.10 -y
conda activate kvcloak
pip install -r requirements.txt
```

### 2. Download Model & Dataset

```bash
# Download model
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ~/model/Llama-3.2-1B

# Dataset already included in repo
ls dataset/
# alpaca_1k.jsonl  gsm8k_1k.jsonl  lmsys-chat-1m_1k.jsonl
```

### 3. Run Complete Workflow

```bash
# Generate KV-cache (20 samples for quick test)
python inference/get_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float32 \
  --device cuda:0 \
  --max-samples 20

# Run attacks on unprotected cache
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type origin \
  --run-injection \
  --start-index 0 --end-index 10

# Apply KV-Cloak protection
python defense/core/kvcloak.py \
  --model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float32 \
  --device cuda:0

# Run attacks on protected cache
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type kvcloak \
  --run-injection \
  --start-index 0 --end-index 10
```

See the difference? Attack accuracy drops from **62.5%** to **9.5%** BERTScore!

---

## Repository Structure

```
kvcloak/
├── attack/              # Attack implementations
│   ├── attacks.py       # Main attack orchestrator
│   ├── inversion.py     # Inversion attack
│   ├── collision.py     # Collision attack
│   └── injection.py     # Injection attack
├── defense/             # Protection methods
│   ├── core/            # KV-Cloak (our method)
│   │   ├── kvcloak.py   # Core protection logic
│   │   └── fusion.py    # Operator fusion optimization
│   ├── baseline/        # Baseline methods
│   │   ├── aes_kvcache.py
│   │   ├── dp_kvcache.py
│   │   └── kvshield.py
│   ├── config/          # Configuration generation
│   └── eval/            # Evaluation scripts
├── inference/           # KV-cache generation
│   ├── get_kvcache.py   # Batch generation
│   └── pdsplit.py       # Prefill-decode split
├── dataset/             # Sample datasets
├── src/                 # Shared utilities
│   ├── config.py        # Central configuration
│   └── security_utils.py # Path validation
└── tests/               # Unit tests
```

---

## Detailed Usage

### Step 1: Generate KV-cache for Theta Configuration

KV-Cloak requires a theta configuration generated from a specific long-sequence sample ("The Bitter Lesson"):

```bash
python inference/pdsplit.py \
  --model_path ~/model/Llama-3.2-1B \
  --device cuda:0 \
  --dtype float32
```

Note the output path: `cache/float32/config/Llama-3.2-1B/<input_hash>/`

### Step 2: Generate Configurations

```bash
# Generate theta config
python defense/config/get_theta_config.py \
  --model-name Llama-3.2-1B \
  --dtype float32 \
  --cache-path cache/float32/config/Llama-3.2-1B/<input_hash>/origin/past_key_values.pt

# Generate KV-Cloak config
python defense/config/get_kvcloak_config.py \
  --model-name Llama-3.2-1B \
  --block-size 16 --S-ratio 1 --M-ratio 1 --theta-ratio 2
```

### Step 3: Full Attack Evaluation

```bash
# All attacks on origin
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type origin \
  --dtype float32 \
  --device cuda:0 \
  --run-inversion --run-collision --run-injection

# All attacks on KV-Cloak protected
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type kvcloak \
  --dtype float32 \
  --device cuda:0 \
  --run-inversion --run-collision --run-injection
```

### Step 4: Performance Benchmarking

```bash
# Micro-benchmark (protection overhead)
python defense/eval/micro_benchmark.py \
  --model-name Llama-3.2-1B \
  --device cuda:0 \
  --dtype float32 \
  --num-trials 5 \
  --max-seq-len 512
```

---

## Advanced Topics

### Collision+ (Chosen-Plaintext-Assisted)

Enhanced collision attack using known plaintext:

```bash
# Generate collision+ config (already included for Llama-3.2-1B)
python attack/get_collision_threshold.py \
  --model_path ~/model/Llama-3.2-1B \
  --target_data_path cache/float32/config/Llama-3.2-1B/<input_hash>/origin/past_key_values.pt \
  --device cuda:0 --dtype float32

# Run with --enhance flag
python attack/attacks.py ... --run-collision --enhance
```

### Configuration Options

Edit `src/config.py` to customize:

```python
# Add new model
MODEL_CONFIGS["My-Model"] = ["My-Model", 256, 256, 3]

# Adjust KV-Cloak parameters
KVCLOAK_DEFAULTS["block_size"] = 32
```

### Security Utilities

```python
from src.security_utils import validate_path, validate_model_name

# Validate user input paths
safe_path = validate_path(user_path, base_dir="./cache")
safe_name = validate_model_name("Llama-3.2-1B")
```

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Test coverage:
- `test_security_utils.py` - Path validation
- `test_config.py` - Configuration management  
- `test_aes_kvcache.py` - AES encryption

---

## Citation

```bibtex
@inproceedings{luo2026shadow,
  title={Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference},
  author={Luo, Zhifan and Shao, Shuo and Zhang, Su and Zhou, Lijing and Hu, Yuke and Zhao, Chenxu and Liu, Zhihao and Qin, Zhan},
  booktitle={Network and Distributed System Security Symposium},
  year={2026}
}
```

---

## License

This repository is released under the Apache-2.0 License. See `LICENSE`.
For dataset usage cautions, see `DATA_POLICY.md`.

---

## Acknowledgments

- This work was published at NDSS 2026
- Thanks to the open-source community for PyTorch and Transformers
- Sample datasets derived from publicly available sources

## Contact

For questions or issues, please open a GitHub Issue.

---

**Privacy Tip**: Always validate user inputs when deploying this system in production. See `src/security_utils.py` for utilities.
