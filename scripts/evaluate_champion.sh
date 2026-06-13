#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-results/champion/config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/champion}"

"${PYTHON_BIN}" - <<'PY'
import json
import os
import runpy
import shutil
from pathlib import Path


OPTIONAL_NUMERIC_NONE = {
    "ENSEMBLE_MIN_SELECTION_SCORE",
    "MAX_CASH_WEIGHT",
    "MAX_WEIGHT",
}


def as_env_value(key, value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        if key in OPTIONAL_NUMERIC_NONE:
            return "none"
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


config_path = Path(os.environ.get("CONFIG_PATH", "results/champion/config.json"))
output_dir = Path(os.environ.get("OUTPUT_DIR", "results/champion"))
if not config_path.exists():
    raise FileNotFoundError(f"Champion config not found: {config_path}")

config = json.loads(config_path.read_text(encoding="utf-8"))
for key, value in config.items():
    env_value = as_env_value(key, value)
    if env_value is not None:
        os.environ[key] = env_value

os.environ["DEVICE"] = os.environ.get("CHAMPION_EVAL_DEVICE", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
output_dir.mkdir(parents=True, exist_ok=True)

runpy.run_module("evaluation.dashboard", run_name="__main__")

for name in [
    "dashboard_benchmark.png",
    "dashboard_metrics.csv",
    "dashboard_cost_scenarios.csv",
]:
    source = Path("assets") / name
    if source.exists():
        shutil.copyfile(source, output_dir / name)

print(f"Champion evaluation artifacts refreshed in {output_dir}")
PY
