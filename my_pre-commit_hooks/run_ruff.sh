# ~/my-hooks/run_ruff.sh
#!/bin/bash
ruff check --force-exclude "$@"
