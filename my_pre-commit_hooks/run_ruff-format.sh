# ~/my-hooks/run_ruff-format.sh
#!/bin/bash

ruff format --config "$HOME/.config/ruff/ruff.toml" --force-exclude "$@"


