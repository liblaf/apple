import ".config/copier/python.just"

default: gen-init lint

bench:
  pytest --benchmark-only --numprocesses=0
