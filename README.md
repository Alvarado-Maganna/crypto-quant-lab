# Crypto Quant Lab

Rigorous, reproducible research on crypto trading strategies.

The goal is to learn quantitative trading and applied data science properly: every strategy starts as an explicit hypothesis, is tested out-of-sample with realistic costs, and is compared against simple benchmarks. **Failed strategies are documented as carefully as successful ones** — an honest negative result is worth more than an inflated backtest.

> This is research and education, not financial advice.

## Structure

```
crypto-quant-lab/
├── data/          # Raw and cached data (gitignored, never committed)
├── src/           # Reusable Python code (data loaders, backtester, metrics)
├── notebooks/     # Exploratory and teaching notebooks, numbered in order
├── experiments/   # One write-up per strategy: hypothesis, data, method, out-of-sample results, conclusion
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
