# CLAUDE.md — Crypto Quant Lab

Instructions for Claude Code in this repository. Read this file at the start of every session.

## Who you are working with

- The owner is a Data Science student and a **beginner** in Python, statistics and time series.
- The goal is deep understanding, not just working code. **Explain every change in simple terms before committing.**
- Conversation with the owner may be in Spanish. **Everything in the repo (code, comments, docstrings, notebooks, READMEs, commit messages, PR descriptions) is in English.**

## Lab rules (non-negotiable)

1. **Hypothesis before code.** Every strategy starts with: "I believe X happens because of economic/behavioral reason Y."
2. **Real costs, always:** maker/taker fees, spread, slippage, and funding on perpetuals.
3. **Out-of-sample or it doesn't count:** walk-forward, purged cross-validation with embargo, and a final test period that is not touched until the end.
4. **Hunt biases:** look-ahead, survivorship, data snooping / overfitting from testing many variants (use the Deflated Sharpe Ratio when relevant).
5. **Always compare against simple benchmarks:** buy & hold, a simple model, and random selection. A complex model must beat the simple ones.
6. **Log everything, including failures,** in `experiments/` (one file per strategy, see `experiments/TEMPLATE.md`).
7. **Ladder to real money:** backtest → live paper trading → small capital that can be lost → scale only with evidence.
8. **Risk before return:** position sizing, max drawdown limit, and a kill switch.

## Data conventions

- All timestamps in **UTC**. Always state the unit when parsing Unix timestamps (`unit="s"` or `unit="ms"`).
- Never use information from a bar (high/low/close) to make a decision at that bar's open (look-ahead bias).
- A price always belongs to a specific exchange and quote currency (e.g. BTC/USDT on exchange X). There is no single "BTC price".
- Raw and cached data go in `data/` (gitignored). Never commit data files.

## Security

- API keys live only in `.env` (gitignored). **Never** commit secrets, and never print them.
- Exchange API keys are created **without withdrawal permission** (read-only or trading only, depending on the phase).
- No bot trades real money without passing the ladder in rule 7 and without a kill switch.

## Git workflow

- **Never commit directly to `main`** (the only exception is the initial commit).
- One branch per feature or experiment: `feat/...`, `exp/...`, `fix/...`, `docs/...`, `setup/...`.
- Small, focused commits with clear messages.
- Open a pull request into `main`. The owner must understand and approve every PR before merging.

## Notebook cycle

Every notebook follows these steps, in order:

1. **Implement:** the code in `src/` with tests in `tests/`, and the notebook in `notebooks/`.
2. **Run:** the notebook top to bottom on a clean kernel; sanity-check the outputs (orders of magnitude, dates, units).
3. **Guided reading of outputs (in Spanish):** go section by section. After each table or plot, explain what it shows (how to read it: axes, units, each line or column), why it comes out that way (the mechanism), and which concept of the current module it connects to. At the end, ask the owner: "What surprised you?"
4. **Reviewer subagent:** run it (see Review below), then fix what it finds.
5. **PR:** clear commits in English, push the branch, open the PR with `gh`.
6. **The owner approves the merge.** Never merge yourself.

Each notebook ends with a **"Key takeaways"** section: 3–5 conclusions, each backed by a number in the notebook.

## Review

- Before a PR is approved, run the `reviewer` subagent (`.claude/agents/reviewer.md`). It checks code quality **and** financial mistakes: look-ahead, missing costs, overfitting, data leakage, timezone/unit errors.

## Stack (tentative)

Python, pandas/polars, numpy, matplotlib, statsmodels, scikit-learn, LightGBM, ccxt (exchange data), Parquet/DuckDB (storage), own backtester (vectorbt only for comparison).
Do not install new packages without explaining what they are for and getting confirmation.
