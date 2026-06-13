import argparse
import json
from pathlib import Path

import pandas as pd


def aggregate_runs(runs_dir="runs", manifest=None):
    rows = []
    rows_by_manifest_key = {}
    manifest_keys = _manifest_keys(manifest) if manifest else None
    for run_dir in sorted(Path(runs_dir).glob("run_*")):
        metrics_path = run_dir / "metrics.jsonl"
        config_path = run_dir / "config.json"
        if not metrics_path.exists():
            continue

        config = _read_json(config_path) if config_path.exists() else {}
        validation_records, train_records = _read_metric_records(metrics_path)

        best_validation = _best_validation(validation_records)
        final_validation = validation_records[-1] if validation_records else None
        final_train = train_records[-1] if train_records else None

        row = {
            "run_id": run_dir.name,
            "_run_mtime": run_dir.stat().st_mtime,
            "seed": config.get("SEED"),
            "ablation": config.get("ABLATION", "unknown"),
            "use_ipm": config.get("USE_IPM"),
            "use_bcm": config.get("USE_BCM"),
            "use_dam": config.get("USE_DAM"),
            "use_arb": config.get("USE_ARB"),
            "use_sparse_network": config.get("USE_SPARSE_NETWORK"),
            "use_online_ipm": config.get("USE_ONLINE_IPM"),
            "lr_actor": config.get("LR_ACTOR"),
            "lr_critic": config.get("LR_CRITIC"),
            "use_target_policy_eval": config.get("USE_TARGET_POLICY_EVAL"),
            "stable_policy_selection_version": config.get("STABLE_POLICY_SELECTION_VERSION"),
            "bcm_lambda": config.get("BCM_LAMBDA"),
            "oracle_anneal_episodes": config.get("ORACLE_ANNEAL_EPISODES"),
            "trading_cost_bps": config.get("TRADING_COST_BPS"),
            "slippage_bps": config.get("SLIPPAGE_BPS"),
            "drawdown_penalty": config.get("DRAWDOWN_PENALTY"),
            "max_cash_weight": config.get("MAX_CASH_WEIGHT"),
            "train_start_date": config.get("TRAIN_START_DATE"),
            "train_end_date": config.get("TRAIN_END_DATE"),
            "valid_start_date": config.get("VALID_START_DATE"),
            "valid_end_date": config.get("VALID_END_DATE"),
            "test_start_date": config.get("TEST_START_DATE"),
            "test_end_date": config.get("TEST_END_DATE"),
        }
        row.update(_prefix_metrics("best_val", best_validation))
        row.update(_prefix_metrics("final_val", final_validation))
        row.update(_prefix_metrics("final_train", final_train))

        if manifest_keys is not None:
            key = _run_key(row)
            if key not in manifest_keys:
                continue
            existing = rows_by_manifest_key.get(key)
            if existing is None or row["_run_mtime"] > existing["_run_mtime"]:
                rows_by_manifest_key[key] = row
            continue

        rows.append(row)

    if manifest_keys is not None:
        rows = list(rows_by_manifest_key.values())

    for row in rows:
        row.pop("_run_mtime", None)
    return pd.DataFrame(rows)


def write_aggregate(runs_dir="runs", output="runs/experiment_summary.csv", manifest=None):
    summary = aggregate_runs(runs_dir, manifest=manifest)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return output_path, summary


def _best_validation(records):
    if not records:
        return None

    return max(
        records,
        key=lambda record: record.get("metrics", {}).get("Selection Score", float("-inf")),
    )


def _prefix_metrics(prefix, record):
    if record is None:
        return {}

    metrics = record.get("metrics", {})
    output = {f"{prefix}_step": record.get("step")}
    for key, value in metrics.items():
        output[f"{prefix}_{_slug(key)}"] = value
    return output


def _read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_metric_records(metrics_path):
    validation_records = []
    train_records = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("stage") == "validation":
                validation_records.append(record)
            elif record.get("stage") == "train_episode":
                train_records.append(record)
    return validation_records, train_records


def _manifest_keys(manifest):
    with Path(manifest).open(encoding="utf-8") as handle:
        runs = json.load(handle)
    return {_run_key(run) for run in runs}


def _run_key(record):
    return (
        int(record["seed"]),
        record["ablation"],
        record["train_start_date"],
        record["train_end_date"],
        record["valid_start_date"],
        record["valid_end_date"],
        record["test_start_date"],
        record["test_end_date"],
    )


def _slug(value):
    return str(value).lower().replace(" ", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser(description="Aggregate AlphaPortfolioRL experiment metrics into one CSV.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="runs/experiment_summary.csv")
    parser.add_argument("--manifest", default=None, help="Optional manifest used to filter runs.")
    args = parser.parse_args()

    output_path, summary = write_aggregate(args.runs_dir, args.output, manifest=args.manifest)
    print(f"Wrote {len(summary)} runs to {output_path}")


if __name__ == "__main__":
    main()
