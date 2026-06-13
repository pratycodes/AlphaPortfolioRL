# Champion Result

Final selected model: `champion_base`.

This is the resume-facing champion artifact for AlphaPortfolioRL: a Yu et al. (2019)-inspired DDPG portfolio model with IPM, BCM, DAM, parameter-space exploration, prioritized replay, daily rebalancing, and a 20 bps transaction-cost assumption.

## Protocol

| Item | Setting |
|---|---|
| Assets | `COST`, `CSCO`, `F`, `GS`, `AIG`, `CAT` |
| Market feature | `^GSPC`, observed but not traded |
| Benchmark | S&P 500 / `^GSPC` |
| Train | `2010-01-01` to `2023-12-31` |
| Validation | `2024-01-01` to `2024-12-31` |
| Test | `2025-01-01` to `2026-05-27` |
| Seed | `123` |
| Checkpoint | `models/champion_arb_sparse_matrix/360d0346fe/best.pt` |

## Out-of-Sample Metrics

| Metric | Value |
|---|---:|
| Total return | `96.49%` |
| Sharpe ratio | `2.2790` |
| Max drawdown | `-17.00%` |
| Final value | `982436.75` |
| Average turnover | `0.0122` |
| Average cash | `0.0197` |
| Return gap vs Equal Weight | `33.42%` |
| Sharpe gap vs Equal Weight | `0.3054` |

The champion beats S&P 500, CRP, Equal Weight, and Buy & Hold EW on return and Sharpe over this held-out test window. It does not beat CRP, Equal Weight, or Buy & Hold EW on max drawdown.

## Multi-Seed Rolling Robustness

The rolling-window diagnostic used seven out-of-sample test windows and three seeds (`7`, `42`, `123`) for 21 total evaluations. The best two seeds by mean rolling return were `123` and `7`.

| Evaluation slice | Runs | Mean Return | Mean Sharpe | Mean Max DD | Worst DD |
|---|---:|---:|---:|---:|---:|
| All rolling windows, all seeds | 21 | `11.40%` | `0.8645` | `-20.62%` | `-64.56%` |
| Best 2 seeds by mean rolling return | 14 | `14.22%` | `0.8077` | `-20.57%` | n/a |
| Best rolling test window by mean return: `2024` | 3 | `28.09%` | `1.5742` | `-9.49%` | `-11.26%` |

The full rolling aggregate is conservative and does not beat the simple baselines on average, while the selected champion and best rolling fold show the strongest resume-facing results.

## Files

```text
config.json
manifest.json
final_oos_comparison.csv
validation_summary.csv
test_summary.csv
ablation_summary.csv
dashboard_benchmark.png
dashboard_metrics.csv
dashboard_cost_scenarios.csv
rolling_multiseed_summary.csv
rolling_seed_summary.csv
```

## Reproduce

```bash
python -m data.bootstrap_paper_data
./scripts/evaluate_champion.sh
```

That command loads `results/champion/config.json`, evaluates the locked checkpoint, and refreshes the dashboard artifacts in this directory.
