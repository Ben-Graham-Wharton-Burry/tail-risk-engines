# Tail Risk Engines

Which tail-risk model should a family office actually run? This project backtests five VaR and Expected Shortfall engines on a diversified multi-asset portfolio over 26 years of daily data, adds drawdown, dependence and machine-learning layers, and turns the winning engine into a volatility-targeting overlay.

The portfolio is a 40/25/15/10/10 policy book across US equities, US bonds, REITs, gold and cash, rebalanced monthly with 10 bps one-way costs. Data starts in 1996, which is when the REIT proxy begins.

---

## Read this first: known issues

These are stated before the results because several of them affect how the results should be read. The notebooks have not been re-executed since they were found.

**1. Notebook outputs and the committed returns file may not match.** The REIT sleeve was switched between two sources (Vanguard VGSIX and a Ken French REIT series) late in the project. Notebook 11 reports the static portfolio's maximum drawdown as 40.0%; notebook 14, which states it runs on the VGSIX series, reports 34.8%. The outputs of notebooks 01 to 13 were very likely computed on a different REIT series than the `returns.csv` committed here, so re-running them will change some numbers.

**2. The volatility target in the prose is not the one the code selects.** Notebook 11 picks the highest target whose realised drawdown stays inside a 17% budget (a 20% tolerance less a 3-point margin). On the data behind its outputs that rule selects **6%**. The notebook prose, the two schematic figures and notebook 14 all describe an **8%** target with roughly 16% drawdown. In notebook 11's own frontier, 8% produces a 21.3% drawdown, which breaches the tolerance.

**3. The headline engine narrowly fails conditional coverage.** The synthesis notebook says GARCH-FHS passes the joint Christoffersen test at 5%. The committed scorecard has p = 0.048, a narrow fail. Notebook 14 finds the same thing on the VGSIX data (p of about 0.02) and shows that an asymmetric GJR-GARCH filter repairs it. Notebook 08's recommendation text also still refers to two portfolios (a 60/40 and a 30/50/20 book) that were retired before the final version; their forecast files remain in `data/forecasts/`.

**4. Three notebooks were last executed out of order** (02, 08 and 10), so some of their outputs may reflect an intermediate state.

**5. Three unit tests were stale and have been updated.** Two expected the rolling engines to subtract the sample mean, but the engine was later switched to a zero conditional mean (standard for one-day VaR, and documented as a design choice); the third looked for a result key one level higher than `backtest_es` documents. The code was correct in all three cases and only the tests were changed. All 131 tests now pass.

## Results

Port_FO, one-day 99% VaR, 6,485 out-of-sample days from May 2000 to April 2026, 65 breaches expected.

| Engine | Breaches | Breach rate | Kupiec p | Christoffersen cc p | Worst-day shortfall beyond VaR |
|---|---|---|---|---|---|
| Gaussian | 135 | 2.08% | < 0.001 | < 0.001 | 5.7% |
| Student-t, nu = 10 | 119 | 1.84% | < 0.001 | < 0.001 | 5.6% |
| Student-t, nu = 6 | 112 | 1.73% | < 0.001 | < 0.001 | 5.6% |
| Student-t, nu = 4 | 96 | 1.48% | < 0.001 | < 0.001 | 5.5% |
| **GARCH(1,1) + FHS** | **80** | **1.23%** | **0.068** | **0.048** | **1.3%** |

Every static-distribution engine fails, and fattening the tails only partly helps: the Student-t with four degrees of freedom still breaches half again as often as it should, because the problem is volatility clustering, not tail shape. Filtering by a GARCH volatility forecast cuts excess breaches from 70 to 15 and the worst shortfall beyond VaR from 5.7% to 1.3%. Its Expected Shortfall forecasts pass both the McNeil-Frey test (p = 0.80) and the Acerbi-Szekely Z2 test.

**Machine learning did not help.** A gradient-boosting exceedance classifier matched the target breach rate almost exactly (0.99%) but failed the independence test badly (cc p near 1e-8): its breaches arrive in clusters, which is the property a risk engine exists to prevent.

**Dependence is regime-dependent.** A DCC model of the equity-bond pair with a Student-t copula (nu = 6) puts the probability of a joint tail event at about 0.05 in calm periods and 0.18 at the correlation peak.

**The overlay.** Scaling exposure down when forecast volatility exceeds the target, and back up only at month-ends, at the 6% target the code selects:

| | CAGR | Ann. vol | Sharpe | Max drawdown |
|---|---|---|---|---|
| Static policy book | 7.24% | 11.2% | 0.48 | 40.0% |
| Responsive overlay, 6% target | 4.88% | 5.5% | 0.48 | 16.1% |
| Static cash blend, matched drawdown | 4.14% | 4.5% | 0.42 | 16.8% |

The overlay does not improve risk-adjusted return over the policy book. What it does is reach a 16% drawdown more cheaply than simply holding cash: at the same drawdown, a static 40/60 blend of the book and cash earns 0.74 points less a year. These figures carry known issue 1.

**Asymmetric GARCH (notebook 14).** GJR-GARCH-FHS passes conditional coverage (p of about 0.24) where GARCH(1,1)-FHS fails it, with a large and stable leverage term, but at the same volatility target it leaves the realised drawdown essentially unchanged. It improves the measured tail more than the realised path.

## Pipeline

Run the notebooks in order; each reads files written by earlier ones.

| # | Notebook | Content |
|---|---|---|
| 01 | Data selection and collection | Portfolio construction, sleeve proxies, sample window |
| 02 | Stylized facts | Fat tails, volatility clustering, ARCH-LM, Ljung-Box |
| 03 | Phase 1 engines | Rolling Gaussian and Student-t VaR and ES |
| 04 | Backtest and stress | Kupiec, Christoffersen, ES tests, stress-window scorecards |
| 05 | GARCH-FHS | GARCH(1,1) with filtered historical simulation, monthly refit |
| 06 | Drawdowns | Maximum drawdown, time under water, CDaR, Monte Carlo and block-bootstrap MDD |
| 07, 07c | ML challenger | Gradient boosting, regularised logit and quantile regression, head-to-head |
| 08 | Synthesis | Engine ranking and cost of being wrong |
| 09 | Dependence | DCC correlation, Student-t copula tail dependence |
| 10 | Exhibits | Consolidated tables and figures |
| 11 | Responsive overlay | Volatility-target frontier, selection rule, static-cash benchmark, robustness |
| 12 | Robustness | Window-length sensitivity |
| 13 | EWMA benchmark | RiskMetrics as an industry-standard comparison |
| 14 | Asymmetric GARCH | GJR and EGARCH filters, overfitting diagnostics |

`src/` holds the engines and tests (`risk.py`, `garch_fhs.py`, `drawdowns.py`, `dcc_garch.py`, `strategy.py`, `asym_garch.py` and others), all covered by `tests/`.

## Running it

```bash
pip install -r requirements.txt
cd tests && for t in test_*.py; do python "$t"; done    # 131 tests
jupyter lab                                              # notebooks 02 onward
```

## Data

`data/returns.csv` holds the daily sleeve and portfolio returns, and the notebooks from 02 onward run from it. Raw price levels are not included: the gold and REIT series come from sources whose licences restrict redistribution (FRED itself withdrew both the LBMA gold price and the Wilshire indices). `fetch_fo_assets.py` documents how to rebuild them from your own downloads, after which notebook 01 can regenerate everything from scratch.

## Author

High_Magus_of_Risk.
