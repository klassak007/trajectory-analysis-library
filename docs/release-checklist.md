# Release Checklist

Use this checklist before publishing a TAL release candidate. It is intentionally
short: the detailed behavioral contracts live in tests and API docs.

## Required Checks

1. Core and architecture tests pass:
   - `conda run -n tal pytest tests/core -q`
   - `conda run -n tal pytest tests/architecture -q`
2. Package build completes:
   - `conda run -n tal python -m build`
3. Packaging metadata is present and current:
   - version in `pyproject.toml`
   - root `README.md`
   - `LICENSE`
4. Documentation builds:
   - `make -C docs html`

## Release Review

- Confirm public docs do not mention implementation planning labels.
- Confirm new public examples have matching docs-example coverage.
- Confirm generated docs are rebuilt from the source Markdown and docstrings.
