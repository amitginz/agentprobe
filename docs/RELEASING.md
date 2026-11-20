# Releasing

The PyPI distribution is **`agentprobe-evals`** (the name `agentprobe` belongs to an unrelated
project). The import name and the CLI command are still `agentprobe`.

## One-time setup (trusted publishing, no token needed)
1. Create a PyPI account and enable 2FA.
2. PyPI → *Your projects* → *Publishing* → **Add a pending publisher**:
   project `agentprobe-evals`, owner `amitginz`, repository `agentprobe`,
   workflow `publish.yml`, environment `pypi`.
3. GitHub → repository *Settings* → *Environments* → create `pypi` (optionally require your approval).

## Each release
1. Bump `version` in `pyproject.toml` and `__version__` in `src/agentprobe/__init__.py`; commit.
2. Create a GitHub Release whose tag is `v<version>`. The `publish` workflow builds, checks that the
   tag matches the package version, and uploads to PyPI.

To test without publishing: `python -m build && twine check dist/*`, then install the wheel into a
clean virtualenv. A version number can never be re-uploaded to PyPI, so get it right first.
