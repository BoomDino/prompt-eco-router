# Contributing

Thanks for helping improve Prompt Decomposer Eco-Router.

## Development setup

Use Python 3.12 and create an isolated environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On fish, activate with `source .venv/bin/activate.fish`. On Windows PowerShell, use `.venv\Scripts\Activate.ps1`.

## Before opening a pull request

Run the same checks as CI:

```bash
python -m compileall -q main.py ui.py eco_router scripts tests
python -m unittest discover -s tests -v
```

Keep tests independent of external services. Do not commit `.env` files, API keys, generated caches, or local virtual environments. For behavior changes, update the README and add or adjust focused tests. Explain any changes to the illustrative energy or price assumptions; these figures are not physical measurements or billing quotes.
