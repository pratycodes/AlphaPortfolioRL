import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field

from config.settings import config


ABLATION_OVERRIDES = {
    "current_baseline": {
        "USE_IPM": "true",
        "USE_ONLINE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "no_bcm": {
        "USE_IPM": "true",
        "USE_ONLINE_IPM": "true",
        "USE_BCM": "false",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "no_ipm": {
        "USE_IPM": "false",
        "USE_ONLINE_IPM": "false",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "no_ipm_no_bcm": {
        "USE_IPM": "false",
        "USE_ONLINE_IPM": "false",
        "USE_BCM": "false",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "paper_costs": {
        "USE_IPM": "true",
        "USE_ONLINE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
        "TRADING_COST_BPS": "5.0",
        "SLIPPAGE_BPS": "0.0",
    },
    "low_cash": {
        "USE_IPM": "true",
        "USE_ONLINE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
        "MAX_CASH_WEIGHT": "0.10",
    },
    "free_cash": {
        "USE_IPM": "true",
        "USE_ONLINE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
        "MAX_CASH_WEIGHT": "none",
    },
    "actor_lr_3e_6": {
        "USE_IPM": "true",
        "USE_ONLINE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
        "LR_ACTOR": "3e-6",
    },
    "actor_lr_3e_5": {
        "USE_IPM": "true",
        "USE_ONLINE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
        "LR_ACTOR": "3e-5",
    },
    "paper_baseline": {
        "USE_IPM": "false",
        "USE_BCM": "false",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "paper_ipm": {
        "USE_IPM": "true",
        "USE_BCM": "false",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "paper_dam": {
        "USE_IPM": "false",
        "USE_BCM": "false",
        "USE_DAM": "true",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "paper_bcm": {
        "USE_IPM": "false",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "paper_ipm_dam": {
        "USE_IPM": "true",
        "USE_BCM": "false",
        "USE_DAM": "true",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "paper_ipm_bcm": {
        "USE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "paper_dam_bcm": {
        "USE_IPM": "false",
        "USE_BCM": "true",
        "USE_DAM": "true",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "paper_all": {
        "USE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "true",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
        "USE_PRIORITIZED_REPLAY": "true",
    },
    "ddpg_only": {
        "USE_IPM": "false",
        "USE_BCM": "false",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
    },
    "ddpg_ipm": {
        "USE_IPM": "true",
        "USE_BCM": "false",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
    },
    "ddpg_bcm": {
        "USE_IPM": "false",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
    },
    "ddpg_ipm_bcm": {
        "USE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
    },
    "ddpg_ipm_bcm_arb": {
        "USE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "true",
        "USE_SPARSE_NETWORK": "false",
    },
    "ddpg_ipm_bcm_sparse": {
        "USE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "true",
    },
    "ddpg_ipm_bcm_arb_sparse": {
        "USE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "false",
        "USE_ARB": "true",
        "USE_SPARSE_NETWORK": "true",
    },
    "ddpg_ipm_bcm_dam": {
        "USE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "true",
        "USE_ARB": "false",
        "USE_SPARSE_NETWORK": "false",
    },
    "ddpg_ipm_bcm_dam_arb_sparse": {
        "USE_IPM": "true",
        "USE_BCM": "true",
        "USE_DAM": "true",
        "USE_ARB": "true",
        "USE_SPARSE_NETWORK": "true",
    },
}


FOCUSED_ABLATIONS = [
    "current_baseline",
    "no_bcm",
    "no_ipm",
    "no_ipm_no_bcm",
    "paper_costs",
    "low_cash",
    "free_cash",
    "actor_lr_3e_6",
    "actor_lr_3e_5",
]

FOCUSED_SEEDS = [42, 7, 123]

FOCUSED_SMOKE_SEEDS = [42]

FOCUSED_SMOKE_SPLIT = {
    "train_start_date": "2018-01-01",
    "train_end_date": "2021-12-31",
    "valid_start_date": "2022-01-01",
    "valid_end_date": "2022-12-31",
    "test_start_date": "2023-01-01",
    "test_end_date": "2023-12-31",
}

FOCUSED_SMOKE_OVERRIDES = {
    "EPISODES": "40",
    "EPISODE_LENGTH": "260",
    "BATCH_SIZE": "64",
    "BUFFER_SIZE": "500",
    "IPM_PRETRAIN_EPOCHS": "15",
    "MIN_SAVE_EPISODE": "10",
    "EARLY_STOPPING_PATIENCE": "2",
    "VALIDATION_FREQ": "5",
}


@dataclass
class ExperimentRun:
    seed: int
    ablation: str
    train_start_date: str
    train_end_date: str
    valid_start_date: str
    valid_end_date: str
    test_start_date: str
    test_end_date: str
    extra_overrides: dict[str, str] = field(default_factory=dict)

    def env(self):
        overrides = {
            "SEED": str(self.seed),
            "TRAIN_START_DATE": self.train_start_date,
            "TRAIN_END_DATE": self.train_end_date,
            "VALID_START_DATE": self.valid_start_date,
            "VALID_END_DATE": self.valid_end_date,
            "TEST_START_DATE": self.test_start_date,
            "TEST_END_DATE": self.test_end_date,
            "ABLATION": self.ablation,
        }
        overrides.update(ABLATION_OVERRIDES[self.ablation])
        overrides.update(self.extra_overrides)
        return overrides


def _date(year, month, day):
    return f"{year:04d}-{month:02d}-{day:02d}"


def walk_forward_splits(cfg=config):
    splits = []
    start = cfg.WALK_FORWARD_START_YEAR
    while True:
        train_start = start
        train_end = train_start + cfg.WALK_FORWARD_TRAIN_YEARS - 1
        valid_start = train_end + 1
        valid_end = valid_start + cfg.WALK_FORWARD_VALID_YEARS - 1
        test_start = valid_end + 1
        test_end = test_start + cfg.WALK_FORWARD_TEST_YEARS - 1

        if test_end > cfg.WALK_FORWARD_END_YEAR:
            break

        splits.append(
            {
                "train_start_date": _date(train_start, 1, 1),
                "train_end_date": _date(train_end, 12, 31),
                "valid_start_date": _date(valid_start, 1, 1),
                "valid_end_date": _date(valid_end, 12, 31),
                "test_start_date": _date(test_start, 1, 1),
                "test_end_date": _date(test_end, 12, 31),
            }
        )
        start += 1

    return splits


def build_runs(cfg=config):
    runs = []
    for split in walk_forward_splits(cfg):
        for ablation in cfg.ABLATIONS:
            if ablation not in ABLATION_OVERRIDES:
                raise ValueError(f"Unknown ablation: {ablation}")
            for seed in cfg.EXPERIMENT_SEEDS:
                runs.append(ExperimentRun(seed=seed, ablation=ablation, **split))
    return runs


def current_research_split(cfg=config):
    return {
        "train_start_date": cfg.TRAIN_START_DATE,
        "train_end_date": cfg.TRAIN_END_DATE,
        "valid_start_date": cfg.VALID_START_DATE,
        "valid_end_date": cfg.VALID_END_DATE,
        "test_start_date": cfg.TEST_START_DATE,
        "test_end_date": cfg.TEST_END_DATE,
    }


def build_focused_runs(cfg=config, smoke=False):
    runs = []
    split = FOCUSED_SMOKE_SPLIT if smoke else current_research_split(cfg)
    seeds = FOCUSED_SMOKE_SEEDS if smoke else FOCUSED_SEEDS
    extra_overrides = FOCUSED_SMOKE_OVERRIDES if smoke else {}
    for ablation in FOCUSED_ABLATIONS:
        if ablation not in ABLATION_OVERRIDES:
            raise ValueError(f"Unknown focused ablation: {ablation}")
        for seed in seeds:
            runs.append(
                ExperimentRun(
                    seed=seed,
                    ablation=ablation,
                    extra_overrides=extra_overrides,
                    **split,
                )
            )
    return runs


def write_manifest(runs, path):
    payload = [asdict(run) for run in runs]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def execute_run(run):
    env = os.environ.copy()
    env.update(run.env())
    subprocess.run([sys.executable, "main.py"], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build or execute AlphaPortfolioRL experiment matrices.")
    parser.add_argument("--execute", action="store_true", help="Actually launch training jobs. Omit for a dry-run manifest only.")
    parser.add_argument("--focused", action="store_true", help="Use the focused tuning matrix for the current research split.")
    parser.add_argument("--smoke", action="store_true", help="Use the short one-seed focused smoke matrix. Implies --focused.")
    parser.add_argument("--manifest", default=None, help="Where to write the run manifest.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for smoke-testing the matrix.")
    args = parser.parse_args()

    runs = build_focused_runs(smoke=args.smoke) if args.focused or args.smoke else build_runs()
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    manifest = args.manifest
    if manifest is None:
        if args.smoke:
            manifest = "runs/focused_smoke_experiment_matrix.json"
        elif args.focused:
            manifest = "runs/focused_experiment_matrix.json"
        else:
            manifest = "runs/experiment_matrix.json"
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    write_manifest(runs, manifest)
    print(f"Wrote {len(runs)} planned runs to {manifest}")

    if args.execute:
        for run in runs:
            execute_run(run)


if __name__ == "__main__":
    main()
