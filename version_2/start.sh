#!/bin/zsh
# Startup script for ragex — sets macOS OMP flags before Python loads
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

exec ragex/bin/uvicorn app.api:app --reload --port 8000 "$@"
