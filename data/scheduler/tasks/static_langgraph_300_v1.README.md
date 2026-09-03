# TradingAgents Static LangGraph 300 Task Seeds

## Summary

This dataset contains 300 point-in-time task seeds for collecting original Static LangGraph trajectories. A record identifies one stock and one historical decision date; it does not contain a trading answer or an LLM-generated trajectory.

## Files

- `static_langgraph_300_v1.jsonl`: 300 task records.
- `static_langgraph_300_v1.manifest.json`: source snapshot, selection method, split counts, universe metadata, file checksum, and usage boundary.
- `../../../training/scheduler/configs/datasets/static_langgraph_300_v1.json`: frozen ticker, sector, and split plan.

## Variables and units

| Field | Definition | Unit / allowed values |
|---|---|---|
| `task_id` | Stable task identifier | string |
| `ticker` | Yahoo Finance ticker | uppercase string |
| `trade_date` | Information cutoff and decision date | ISO `YYYY-MM-DD` |
| `asset_type` | Instrument type | `stock` |
| `split` | Dataset role | `train`, `validation`, `test` |
| `sector` | Frozen sector stratum | 11 normalized sector names |
| `seed_family` | Market-state sampling stratum | 6 declared families |
| `trailing_return_5d` | Adjusted close return through the decision date | fraction |
| `annualized_volatility_20d` | 20-day return standard deviation × √252 | fraction per year |
| `volume_zscore_20d` | Current volume relative to its trailing 20-day window | dimensionless z-score |

## Methods and provenance

The build plan assigns each ticker to exactly one split. Historical OHLCV and earnings dates are retrieved sequentially from yfinance. Every feature at date `t` uses observations no later than `t`. Six ranked candidate pools are selected independently inside each split, subject to unique ticker/date pairs, a six-task limit per ticker, and a minimum 21-day same-ticker gap.

The exact retrieval time, yfinance package version, task count, selection settings, and SHA-256 checksum are recorded in the manifest. The implementation is in `training/scheduler/build_task_seeds.py` and `training/scheduler/scaled_task_seeds.py`.

## Software and environment

- Python 3.12-compatible project environment.
- pandas as declared by the repository.
- yfinance 1.7.0 for the frozen v1 snapshot.

## Access and licence

The repository stores derived task metadata only. Raw yfinance market data and third-party news are not redistributed. Users rebuilding the task pool or collecting Agent trajectories are responsible for the applicable provider terms. No standalone dataset DOI or separate dataset licence has been assigned.

## Citation

Until a versioned repository release or dataset DOI exists, cite the repository commit used to generate the task pool together with the TradingAgents paper. Do not invent or infer a persistent identifier.

## Current boundary

The 300 Static LangGraph trajectories, SFT examples, human conflict review, and trained Scheduler checkpoint have not yet been generated.
