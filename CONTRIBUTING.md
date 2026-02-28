# Contributing to KV-CLOAK

Thanks for your interest in improving KV-CLOAK.

## Ways to contribute

- Report reproducibility issues
- Improve docs and examples
- Add tests and benchmarks
- Fix bugs and improve compatibility

## Development setup

```bash
conda create --name kvcloak-dev python=3.10 -y
conda activate kvcloak-dev
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black isort
```

## Before submitting a PR

1. Run tests locally:

```bash
pytest tests/ -v
```

2. Run style checks:

```bash
flake8 src/ tests/
black --check src/ tests/
isort --check-only src/ tests/
```

3. Keep changes focused and small.

## Pull request checklist

- [ ] The change is scoped to a clear problem
- [ ] Tests are added or updated
- [ ] Existing tests pass locally
- [ ] README/docs are updated if behavior changed
- [ ] No sensitive data or credentials are included

## Security and dual-use note

This repository includes attack implementations for defensive research. By contributing, you agree to follow `RESPONSIBLE_USE.md` and avoid enabling misuse.

## Issue reporting

When opening issues, please include:

- OS and Python version
- `torch` and `transformers` versions
- Exact command and full traceback
- Minimal steps to reproduce

## Code of conduct

Be respectful and constructive. We welcome good-faith contributions from all backgrounds.
