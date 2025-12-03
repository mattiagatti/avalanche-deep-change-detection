#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_infer.sh /path/to/base_dir [extra args passed to infer.py...]
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/base_dir [infer.py args...]"
  exit 1
fi

BASE_DIR="$1"
shift  # remaining "$@" are forwarded to infer.py

SCRIPT_PATH="infer.py"
VENV_PATH=".venv/bin/activate"

# Activate venv if present
if [[ -f "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
else
  echo "Warning: virtualenv not found at '$VENV_PATH' — continuing without it."
fi

# Ensure BASE_DIR exists and is a directory
if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: '$BASE_DIR' is not a directory."
  exit 1
fi

shopt -s nullglob

found_any=false
for event_dir in "$BASE_DIR"/*/; do
  bn="$(basename "$event_dir")"
  
  # Skip hidden dirs and "preds" folder
  [[ "$bn" == .* ]] && continue
  [[ "$bn" == "preds" ]] && continue

  found_any=true
  echo "Processing: $event_dir"
  python "$SCRIPT_PATH" --event-path "$event_dir" "$@"
done

if ! $found_any; then
  echo "No subfolders found in '$BASE_DIR'."
fi