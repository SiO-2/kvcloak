# KV-CLOAK

Code repository for the NDSS paper:

**Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**  
https://www.ndss-symposium.org/ndss-paper/shadow-in-the-cache-unveiling-and-mitigating-privacy-risks-of-kv-cache-in-llm-inference/

## Repository Layout

- `attack/`: KV-cache inversion/collision/injection attacks.
- `defense/`: KV-Cloak, DP, AES, KV-Shield baselines, and evaluation scripts.
- `inference/`: KV-cache generation utilities.
- `dataset/`: small sampled datasets used for experiments.
- `src/`: Shared utilities including configuration (`config.py`) and security helpers (`security_utils.py`).

## Environment Setup

```bash
conda create --name kvcloak python=3.10 -y
conda activate kvcloak
pip install -r requirements.txt
```

## Model and Dataset Preparation

By default, scripts look for models in `~/model` and datasets in `~/dataset`.

Example downloads:

```bash
huggingface-cli download sentence-transformers/all-mpnet-base-v2 --local-dir ~/model/all-mpnet-base-v2
modelscope download --model LLM-Research/Llama-3.2-1B --local_dir ~/model/Llama-3.2-1B
modelscope download --dataset AI-ModelScope/lmsys-chat-1m --local_dir ~/dataset/lmsys-chat-1m
```

## Quick Start

### 1) Generate KV-cache

```bash
python inference/get_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float32 \
  --device cuda:0
```

This will process the dataset and save KV-cache to `cache/float32/lmsys-chat-1m_1k/Llama-3.2-1B/<input_hash>/origin/`.

### 2) Run attack (`attack/attacks.py`, injection by default)

```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type origin \
  --dtype float32 \
  --device cuda:0
```

**Useful options:**

- `--start-index`, `--end-index`: run attacks on a subset of cache samples (e.g., `--start-index 0 --end-index 10` for quick testing).
- `--run-inversion`, `--run-collision`, `--run-injection/--no-run-injection`: select which attacks to run.
- `--logical-batch-size`, `--micro-batch-size`, `--stop-partition`: collision search controls.
- `--enhance/--no-enhance`: enable chosen-plaintext-assisted threshold selection for collision (uses fixed rank guess `r=8`).

### 3) Run standalone collision script

```bash
python attack/collision.py \
  --model_path ~/model/Llama-3.2-1B/ \
  --target_data_path cache/torch.float32/lmsys-chat-1m_1k/Llama-3.2-1B/<input_hash>/origin/past_key_values.pt \
  --device cuda:7 \
  --dtype float32 \
  --logical_batch_size 256 \
  --micro_batch_size 256 \
  --stop_partition 3 \
  --target_gap 3
```

## KV-Cloak Protection vs. Attack Accuracy

This experiment compares attack accuracy before and after KV-Cloak protection on the same model and dataset.

### Prerequisites

Before running KV-Cloak protection, you need to generate the configuration files.

**Important**: The theta configuration must be generated using the **KV-cache of a specific long-sequence sample** to avoid statistical outliers. According to the paper, use the following text ("The Bitter Lesson"):

> "One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning. The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries."

### Step 0: Generate KV-cache for Theta Configuration

Use `inference/pdsplit.py` to generate the KV-cache for this specific input text (the default input is already set to this text):

```bash
python inference/pdsplit.py \
  --model_path ~/model/Llama-3.2-1B \
  --device cuda:0 \
  --dtype float32
```

This will generate the cache and output its location:
```
cache saved in cache/float32/config/Llama-3.2-1B/<input_hash>/
```

Note down the `<input_hash>` value (e.g., `a1b2c3d4...`) for the next step.

### Step 1: Generate Theta Configuration

Use the generated cache to create the theta configuration:

```bash
python defense/config/get_theta_config.py \
  --model-name Llama-3.2-1B \
  --dtype float32 \
  --cache-path cache/float32/config/Llama-3.2-1B/<input_hash>/origin/past_key_values.pt
```

**Note**: Replace `<input_hash>` with the hash from Step 0. This generates `defense/config/kvcloak/theta/Llama-3.2-1B.json`.

### Step 2: Generate KV-Cloak Configuration

```bash
python defense/config/get_kvcloak_config.py \
  --model-name Llama-3.2-1B \
  --block-size 16 --S-ratio 1 --M-ratio 1 --theta-ratio 2
```

This generates `defense/config/kvcloak/b16_S1_M1_t2/Llama-3.2-1B.pt`.

### Step 3: Generate Origin KV-cache (if not already done)

```bash
python inference/get_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float32 \
  --device cuda:0
```

### Step 4: Run Attacks on Origin KV-cache (Baseline)

**Quick test (10 samples):**
```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type origin \
  --dtype float32 \
  --device cuda:0 \
  --run-injection \
  --start-index 0 \
  --end-index 10
```

**Full evaluation:**
```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type origin \
  --dtype float32 \
  --device cuda:0 \
  --run-inversion \
  --run-collision \
  --run-injection
```

If you want `collision+` (chosen-plaintext-assisted thresholding with fixed `r=8`), you need to first generate the distance distribution configuration using the same "The Bitter Lesson" KV-cache:

**Generate Collision+ Configuration:**

Use the dedicated script that combines statistic and analysis steps:

```bash
python attack/get_collision_threshold.py \
  --model_path ~/model/Llama-3.2-1B \
  --target_data_path cache/float32/config/Llama-3.2-1B/<input_hash>/origin/past_key_values.pt \
  --device cuda:0 \
  --dtype float32
```

This script will:
1. Generate distance statistics for each position
2. Analyze target token vs other tokens distance distributions
3. Save the collision configuration to `attack/config/origin/float32/Llama-3.2-1B.json`

Then run attacks with `--enhance`:

```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type origin \
  --dtype float32 \
  --device cuda:0 \
  --run-collision \
  --enhance
```

### Step 5: Apply KV-Cloak Protection

```bash
python defense/core/kvcloak.py \
  --model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float32 \
  --device cuda:0
```

This command:
- Uses model: `Llama-3.2-1B`
- Uses dataset: `./dataset/lmsys-chat-1m_1k.jsonl`
- Uses config: `defense/config/kvcloak/b16_S1_M1_t2/Llama-3.2-1B.pt`
- Writes protected cache to each sample directory under `kvcloak/past_key_values.pt`

### Step 6: Run Attacks on KV-Cloak-Protected KV-cache

**Quick test (10 samples):**
```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type kvcloak \
  --dtype float32 \
  --device cuda:0 \
  --run-injection \
  --start-index 0 \
  --end-index 10
```

**Full evaluation:**
```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type kvcloak \
  --dtype float32 \
  --device cuda:0 \
  --run-inversion \
  --run-collision \
  --run-injection
```

## Defense Scripts

The `defense/` directory is organized as follows:

```
defense/
├── core/          # Core defense implementations (our method)
│   ├── kvcloak.py       - KV-Cloak protection
│   └── fusion.py        - operator fusion
├── baseline/      # Baseline methods for comparison
│   ├── dp_kvcache.py    - Differential Privacy baseline
│   ├── aes_kvcache.py   - AES encryption baseline
│   └── kvshield.py      - KVShield baseline
```

**Notes:**
- `core/` contains our method (KV-Cloak) and its performance optimization module:
  - `kvcloak.py`: The main KV-Cloak protection from NDSS 2026
  - `fusion.py`: Performance optimization that fuses the M matrix into attention weights during model deployment.
- `baseline/` contains baseline methods for comparison:
  - `dp_kvcache.py`: Differential Privacy baseline
  - `aes_kvcache.py`: AES encryption baseline
  - `kvshield.py`: KVShield baseline from another paper (has safety and compatibility issues)
- When running scripts from subdirectories, use Python's module syntax or adjust the working directory:

```bash
# Option 1: Run from repository root with module path
cd ~/test/kvcache
python defense/eval/mmlu_eval.py ...

# Option 2: Run from defense directory
cd ~/test/kvcache/defense
python -m eval.mmlu_eval ...
```

Example evaluation commands:

```bash
# Generate theta config (prerequisite for KV-Cloak)
python defense/config/get_theta_config.py \
  --model-name Llama-3.2-1B \
  --dtype float32 \
  --cache-path cache/float32/lmsys-chat-1m_1k/Llama-3.2-1B/<input_hash>/origin/past_key_values.pt

# Generate KV-Cloak config
python defense/config/get_kvcloak_config.py \
  --model-name Llama-3.2-1B \
  --block-size 16 --S-ratio 1 --M-ratio 1 --theta-ratio 2

# Apply KV-Cloak protection
python defense/core/kvcloak.py \
  --model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float32 --device cuda:0

# Apply DP protection (baseline)
python defense/baseline/dp_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float32 --device cuda:0

# Generate collision+ (CPA) threshold config
python attack/get_collision_threshold.py \
  --model_path ~/model/Llama-3.2-1B \
  --target_data_path cache/float32/config/Llama-3.2-1B/<input_hash>/origin/past_key_values.pt \
  --device cuda:0 \
  --dtype float32

# Performance benchmark - test protection overhead (micro benchmark)
python defense/eval/micro_benchmark.py \
  --model-name Llama-3.2-1B \
  --device cuda:0 \
  --dtype float32 \
  --num-trials 5 \
  --max-seq-len 512

# Model accuracy evaluation - MMLU benchmark
python defense/eval/mmlu_eval.py \
  --model-name Llama-3.2-1B \
  --device cuda:0 \
  --dtype float32 \
  --protect-type origin

# Model accuracy evaluation - SQuAD benchmark
python defense/eval/squad_eval.py \
  --model-name Llama-3.2-1B \
  --device cuda:7 \
  --dtype float32 \
  --protect-type kvcloak
```

**Note on evaluation types:**
- **micro_benchmark**: Tests protection overhead (latency) on synthetic KV-cache data
- **mmlu_eval**: Tests downstream task accuracy on MMLU dataset with protected KV-cache
- **squad_eval**: Tests downstream task accuracy on SQuAD dataset with protected KV-cache

For all scripts under `defense/`, run `python defense/<script>.py --help` to see full CLI options.

## Reproducibility Notes

- Most scripts set `torch.manual_seed(42)` for deterministic behavior.
- Intermediate cache/results are not committed by default (`cache/`, `attack/result/`, `defense/result/`, `outputs/`).
- Run a syntax smoke check before large experiments:

```bash
python -m compileall attack defense inference
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{luo2026shadow,
  title={Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference},
  author={Luo, Zhifan and Shao, Shuo and Zhang, Su and Zhou, Lijing and Hu, Yuke and Zhao, Chenxu and Liu, Zhihao and Qin, Zhan},
  booktitle={Network and Distributed System Security Symposium (NDSS)},
  year={2026},
  publisher={Internet Society},
  url={https://www.ndss-symposium.org/ndss-paper/shadow-in-the-cache-unveiling-and-mitigating-privacy-risks-of-kv-cache-in-llm-inference/}
}
```

## Testing

The repository includes unit tests in the `tests/` directory. To run the tests:

```bash
# Install pytest and test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_security_utils.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Coverage

Current test modules:
- `test_security_utils.py` - Security validation functions
- `test_config.py` - Configuration management
- `test_aes_kvcache.py` - AES encryption/decryption

## Configuration

You can customize the project settings by editing `src/config.py`:

- **Model paths**: Set `MODEL_ROOT` and `DATASET_ROOT`
- **Model configurations**: Add new models to `MODEL_CONFIGS` dictionary
- **Default parameters**: Adjust `KVCLOAK_DEFAULTS`, `DP_DEFAULTS`, `ATTACK_DEFAULTS`
- **The Bitter Lesson text**: Update `BITTER_LESSON_TEXT` if needed

Example:
```python
# Add a new model configuration
MODEL_CONFIGS["My-Model"] = ["My-Model", 256, 256, 3]

# Change default block size for KV-Cloak
KVCLOAK_DEFAULTS["block_size"] = 32
```

## Security

The repository includes basic path validation utilities in `src/security_utils.py`. These help prevent:
- Path traversal attacks
- Directory escape attempts
- Malicious model names

To use in your code:
```python
from src.security_utils import validate_path, validate_model_name

# Validate a path
safe_path = validate_path(user_input_path, base_dir="./cache")

# Validate a model name
safe_name = validate_model_name(user_model_name)
```

## License

This repository is released under the Apache-2.0 License. See `LICENSE`.
For dataset usage and redistribution cautions, see `DATA_POLICY.md`.
