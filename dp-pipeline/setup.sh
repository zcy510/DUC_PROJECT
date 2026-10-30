#!/usr/bin/env bash
# Reproducible setup: creates ./.venv with the exact pinned packages,
# checks Python and Java versions, and runs the correctness tests.
#
#   ./setup.sh                      # uses python3 (or $PYTHON if set)
#   PYTHON=python3.12 ./setup.sh    # choose a specific interpreter
#
# Afterwards:  source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

# 1. Python must be 3.11-3.13 (PipelineDP requirement)
if ! command -v "$PY" >/dev/null; then
  echo "ERROR: '$PY' not found. Try: PYTHON=python3.12 bash setup.sh"
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
  echo "ERROR: need Python 3.11-3.13, got: $("$PY" --version 2>&1 || echo 'not found')"
  echo "       Try: PYTHON=python3.12 ./setup.sh   (or: pyenv install 3.12)"
  exit 1
fi
echo "Python: $("$PY" --version)"

# 2. Java must be 17+ (Spark 4.0 requirement)
if ! command -v java >/dev/null; then
  echo "ERROR: Java not found. Install Java 17, e.g. 'brew install openjdk@17'."
  exit 1
fi
JAVA_MAJOR=$(java -version 2>&1 | awk -F'"' '/version/ {split($2, v, "."); print v[1]; exit}')
if [ "${JAVA_MAJOR:-0}" -lt 17 ]; then
  echo "ERROR: need Java 17 or 21, found Java ${JAVA_MAJOR:-unknown}."
  exit 1
fi
echo "Java:   $JAVA_MAJOR"

# 3. Create .venv and install the pinned packages
"$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "Installed: $(.venv/bin/python -c 'import pyspark, pipeline_dp; print("pyspark", pyspark.__version__)')"

# 4. Prove the environment works: T0 correctness tests
.venv/bin/python -m pytest -q

echo
echo "Done. Activate the environment with:  source .venv/bin/activate"
