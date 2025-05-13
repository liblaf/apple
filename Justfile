set allow-duplicate-recipes := true

import ".config/copier/python.just"

default: gen-init lint

test *ARGS:
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
      pytest --benchmark-disable --numprocesses="auto" {{ ARGS }}
