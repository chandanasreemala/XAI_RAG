#!/usr/bin/env bash
set -euo pipefail
# Start the version_2 FastAPI app on port 8000 (default)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PORT=${1:-8000}
echo "Starting version_2 server on port ${PORT} (pwd=$HERE)"
# Activate conda env if available (optional)
if command -v conda >/dev/null 2>&1; then
  CONDA_ENV=${CONDA_DEFAULT_ENV:-ragex}
  if [[ -n "$CONDA_ENV" ]]; then
    echo "Activating conda env: $CONDA_ENV"
    # shellcheck disable=SC1091
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || true
  fi
fi

# WATCHFILES_FORCE_POLLING bypasses inotify entirely (uses polling instead),
# avoiding "OS file watch limit reached" on shared servers with low inotify limits.
exec env WATCHFILES_FORCE_POLLING=true uvicorn app.api:app --reload --reload-dir app --port "$PORT" --host 0.0.0.0
