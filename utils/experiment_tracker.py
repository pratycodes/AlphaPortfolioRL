import json
import subprocess
from datetime import datetime
from pathlib import Path


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class ExperimentTracker:
    def __init__(self, config, run_name=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_name or f"run_{timestamp}_seed_{config.SEED}"
        self.run_dir = Path(config.EXPERIMENT_DIR) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def write_manifest(self, config):
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": self._git_commit(),
            "config": self._config_dict(config),
        }
        self._write_json(self.run_dir / "manifest.json", manifest)
        self._write_json(self.run_dir / "config.json", manifest["config"])

    def log_metrics(self, stage, step, metrics):
        record = {
            "stage": stage,
            "step": step,
            "metrics": metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")

    def _write_json(self, path, payload):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=_json_default, indent=2, sort_keys=True)

    def _config_dict(self, config):
        if hasattr(config, "model_dump"):
            return config.model_dump()
        return dict(config.__dict__)

    def _git_commit(self):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None
