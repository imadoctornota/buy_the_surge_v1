# Should You Buy a Stock After It Surges?

Code for the analysis behind my video testing what happens after a stock makes an unusually large move in a single day.

The focus universe is **AAPL, MSFT, TSLA, NVDA, AMZN, META, and NFLX**, with **SPY** used as the market benchmark.

## Main question

After a stock has already surged, are future returns worse because the move has been "missed," or does the strength tend to continue?

The primary test looks at first crossings of **5%, 10%, 15%, and 20% one-day returns**. The signal is observed at the close on day *t*, and entry occurs at the close on day *t+1*. Forward returns are measured over 5, 21, 63, and 252 trading days.

Two versions of the outcome are reported:

- **Raw edge:** post-surge return minus the same stock's ordinary forward return.
- **SPY-relative edge:** the same comparison after measuring compounded performance relative to SPY over matching dates.

The pooled result is calculated at the stock level first and then averaged across stocks so that a volatile stock with many events does not automatically dominate the result.

## Results reported in the video

| One-day surge | Usable events | 3-month raw edge | 3-month SPY-relative edge |
| ---: | ---: | ---: | ---: |
| 5% | 1,422 | +0.8 pp | +1.1 pp |
| 10% | 281 | +6.8 pp | +7.0 pp |
| 15% | 98 | +12.0 pp | +11.8 pp |
| 20% | 34 | +21.4 pp | +20.8 pp |

The smaller 5% move showed little advantage in this sample. Larger moves had larger estimated edges, but the number of observations dropped quickly. A separate portfolio simulation also showed why an event-study edge does not automatically translate into a superior strategy: waiting for rare signals leaves a large share of capital uninvested.

## Additional checks

The code includes:

- a 21-trading-day cooldown between events;
- mean-versus-median comparisons;
- a two-stage bootstrap that resamples stocks and then events within stocks;
- alternative 1-day, 5-day, and 21-day surge definitions;
- a fixed-slot portfolio simulation with a simple transaction-cost assumption.

## Running the analysis

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py
```

The sample is frozen through **September 15, 2026**. `yfinance` downloads the required adjusted-close history when the script runs.

## Data handling

This clean version intentionally contains **no cached Yahoo Finance price history and no event-level CSV exports**. Prices are downloaded at runtime, used in memory, and discarded when the process ends. The code writes only aggregate summary tables and figures locally; generated directories and common local data formats are excluded by `.gitignore`.

## Repository layout

```text
.
├── run_analysis.py
├── src/
│   ├── surge_study.py
│   ├── portfolio.py
│   └── plots.py
├── requirements.txt
└── .gitignore
```

## Limitations

The seven-stock universe was selected with hindsight and consists of prominent survivors, so survivorship and selection bias are substantial. Events and forward-return windows can overlap. The cooldown and hierarchical bootstrap reduce some dependence concerns but do not make all observations independent. The portfolio simulation is deliberately simplified and does not model taxes, variable spreads, market impact, or execution uncertainty.

A stronger market-wide test would use a historical, delisting-aware universe and a genuinely untouched validation sample.

## Disclaimer

For educational and informational purposes only. Nothing here is financial or investment advice.
