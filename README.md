# Crypto Quant Lab

Rigorous, reproducible research on crypto trading strategies. (usign AI)

The goal is to learn quantitative trading and applied data science properly: every strategy starts as an explicit hypothesis, is tested out-of-sample with realistic costs, and is compared against simple benchmarks. **Failed strategies are documented as carefully as successful ones** — an honest negative result is worth more than an inflated backtest.

> This is research and education, not financial advice.

## Structure

```
crypto-quant-lab/
├── data/          # Raw and cached data (gitignored, never committed)
├── src/           # Reusable Python code (data loaders, backtester, metrics)
├── notebooks/     # Exploratory and teaching notebooks, numbered in order
├── experiments/   # One write-up per strategy: hypothesis, data, method, out-of-sample results, conclusion
├── scripts/       # Long-running command-line tools (e.g. the order-book collector)
├── tests/         # Unit tests
├── docs/          # Longer write-ups (e.g. crypto-primer.md)
├── CLAUDE.md      # Lab rules and conventions for AI-assisted development
└── requirements.txt
```

## Roadmap

| Phase | Topic | Output |
|---|---|---|
| 0 | Crypto fundamentals | 5 small notebooks + `docs/crypto-primer.md` |
| 1 | Markets and trading | First data exploration notebook |
| 2 | Data science for finance | Explanatory notebooks |
| 3 | Model catalog | Catalog document |
| 4 | Simple models | Own backtesting engine + first experiments |
| 5 | Intermediate and advanced models | Documented experiments |
| 6 | Production | Automation, paper trading, monitoring |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts

### `scripts/collect_order_book.py`: 24-hour order-book collector

Takes raw order-book snapshots of Binance spot BTC/USDT, ETH/USDT, one altcoin (default
PENGU/USDT) and the USDⓈ-M perpetual BTC/USDT:USDT, with the same method as notebook 06
(public data, no API key). Default: one round per minute for 24 hours.

```bash
python scripts/collect_order_book.py                                  # 24 h, every 60 s
python scripts/collect_order_book.py --duration-h 2 --out-dir data/ob_test
python scripts/collect_order_book.py --help                           # all options
```

- Output: one Parquet file per UTC hour in `--out-dir` (default `data/order_book_24h/`), plus a
  `collector.log`. Writes are atomic, so stopping it (Ctrl+C) or a crash loses at most the
  round in progress; restarting in the same folder is safe.
- Network errors and rate limits are retried with growing waits; a market that keeps failing is
  skipped for that round and logged (never filled in).
- Keep the computer awake while it runs. Expect roughly 40 million rows per day; load one
  market at a time with `load_snapshots(folder, markets=[...])`.
- Load the data with `from src.data.order_book import load_snapshots`.
