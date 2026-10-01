#!/usr/bin/env bash
# Full benchmark with sensible Pi defaults. Extra args are passed to benchmark.py,
# e.g.  scripts/run_benchmark.sh --repeats 3 --k 5
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
python tests/smoke_test.py
python benchmark.py "$@"
