default: gen-init lint

gen-init:
    ./scripts/gen-init.sh

lint: lint-toml lint-python

lint-toml:
    sort-toml .ruff.toml pyproject.toml

lint-python:
    ruff check --fix
