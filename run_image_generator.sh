#!/usr/bin/env bash

# run_image_generator.sh
# A convenience wrapper to ensure the Python virtual environment is
# set up before running image_generator.py with any arguments.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_EXEC="$VENV_DIR/bin/python"

# 1. Create the virtual environment if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# 2. Activate the virtual environment
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# 3. Ensure pip is up to date
pip install --upgrade pip >/dev/null

# 4. Install/update required dependencies
pip install -r "$PROJECT_DIR/requirements.txt" >/dev/null

# 5. Run the image generator with all passed arguments
exec "$PYTHON_EXEC" "$PROJECT_DIR/image_generator.py" "$@" 