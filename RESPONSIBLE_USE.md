## Responsible Use Policy

This repository contains both defense mechanisms and attack implementations related to KV-cache privacy risks in LLM inference.

### Intended Use

- Academic research and reproducibility.
- Defensive security evaluation and benchmarking.
- Testing in controlled environments with explicit authorization.

### Prohibited Use

- Any unauthorized access, model probing, or data extraction.
- Privacy violations against users, organizations, or systems.
- Any activity that violates laws, regulations, contracts, or platform terms.

### User Responsibilities

- You are solely responsible for how you use this code.
- Ensure you have permission before running attack scripts.
- Prefer anonymized/synthetic data when possible.
- Follow your institution's ethics and disclosure requirements.

### Safety Controls in This Repository

Attack entry scripts require explicit acknowledgment before execution:

- CLI flag: `--i-understand-risks`
- Environment variable: `KVCLOAK_ALLOW_ATTACKS=1`

These controls are not a complete misuse prevention mechanism; they are a friction layer to reduce accidental misuse.

### Disclosure

If you discover a new vulnerability using this repository, we recommend coordinated disclosure to affected vendors/operators before public release of exploit details.
