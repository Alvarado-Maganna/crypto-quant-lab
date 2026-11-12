# Experiment: <name>

- **Date started:** YYYY-MM-DD
- **Status:** planned | running | concluded (edge found / no edge / inconclusive)
- **Branch / PR:** 

## 1. Hypothesis
"I believe X happens because of economic/behavioral reason Y."

## 2. Data
- Source / exchange / symbol / quote currency:
- Frequency and period (UTC):
- In-sample / out-of-sample / final test split:

## 3. Method
- Signal definition (only uses information available at decision time):
- Costs included: maker/taker fees, spread, slippage, funding:
- Validation: walk-forward / purged CV with embargo:
- Number of variants tried (for data-snooping correction):

## 4. Benchmarks
- Buy & hold:
- Simple model:
- Random selection:

## 5. Results (out-of-sample)
| Metric | Strategy | Buy & hold | Simple | Random |
|---|---|---|---|---|
| Return (net of costs) | | | | |
| Sharpe | | | | |
| Max drawdown | | | | |

## 6. Bias checklist
- [ ] No look-ahead
- [ ] No survivorship bias
- [ ] Costs realistic
- [ ] Variants counted; Deflated Sharpe if relevant

## 7. Conclusion
What did we learn? What would falsify it? Next step?
