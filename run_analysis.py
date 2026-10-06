"""Run the stock-surge event study without saving vendor-sourced price history."""

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.plots import save_edge_plot, save_portfolio_plot
from src.portfolio import equal_weight_buy_and_hold, performance_metrics, simulate_fixed_slot_strategy
from src.surge_study import (
    add_forward_returns,
    apply_cooldown,
    detect_surge_events,
    hierarchical_bootstrap,
    ordinary_day_baselines,
    pooled_edge_summary,
)


FOCUS_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
BENCHMARK_TICKER = "SPY"
START_DATE = "2000-01-01"
# yfinance end is exclusive. The original analysis ended September 15, 2026.
END_DATE = "2026-09-16"

SURGE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "1y": 252}
ENTRY_LAG = 1
COOLDOWN_DAYS = 21
MIN_EVENTS_PER_STOCK = 3
BOOTSTRAP_DRAWS = 2000
RANDOM_SEED = 42

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")


def download_prices(tickers):
    raw = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("No price data were returned.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.reindex(columns=tickers).sort_index().dropna(how="all")
    missing = close.columns[close.notna().sum() == 0].tolist()
    if missing:
        raise RuntimeError(f"No usable history returned for: {missing}")
    return close


def alternative_window_summary(prices, market, baseline):
    specs = {1: [0.05, 0.10, 0.15], 5: [0.10, 0.20, 0.30], 21: [0.20, 0.30, 0.50]}
    rows = []
    for window, thresholds in specs.items():
        signal = prices / prices.shift(window) - 1.0
        events = detect_surge_events(signal, thresholds=thresholds, window=window)
        results = add_forward_returns(events, prices, market, HORIZONS, entry_lag=ENTRY_LAG)
        summary = pooled_edge_summary(
            results,
            baseline,
            thresholds=thresholds,
            horizons={"3m": HORIZONS["3m"]},
            min_events_per_stock=MIN_EVENTS_PER_STOCK,
        )
        summary["surge_window_days"] = window
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_prices = download_prices(FOCUS_TICKERS + [BENCHMARK_TICKER])
    prices = all_prices[FOCUS_TICKERS].copy()
    market = all_prices[BENCHMARK_TICKER].copy()

    surge_return = prices.pct_change(1)
    events = detect_surge_events(surge_return, SURGE_THRESHOLDS, window=1)
    results = add_forward_returns(events, prices, market, HORIZONS, entry_lag=ENTRY_LAG)
    baseline = ordinary_day_baselines(prices, market, HORIZONS, entry_lag=ENTRY_LAG)
    pooled = pooled_edge_summary(
        results,
        baseline,
        thresholds=SURGE_THRESHOLDS,
        horizons=HORIZONS,
        min_events_per_stock=MIN_EVENTS_PER_STOCK,
    )

    cooldown_events = apply_cooldown(events, prices, cooldown_days=COOLDOWN_DAYS)
    cooldown_results = add_forward_returns(cooldown_events, prices, market, HORIZONS, entry_lag=ENTRY_LAG)
    cooldown_summary = pooled_edge_summary(
        cooldown_results,
        baseline,
        thresholds=SURGE_THRESHOLDS,
        horizons=HORIZONS,
        min_events_per_stock=MIN_EVENTS_PER_STOCK,
    )

    bootstrap_rows = []
    for outcome in ["raw", "relative"]:
        bootstrap_rows.append(
            hierarchical_bootstrap(
                results,
                baseline,
                threshold=0.10,
                horizon="3m",
                outcome=outcome,
                draws=BOOTSTRAP_DRAWS,
                seed=RANDOM_SEED,
                min_events_per_stock=MIN_EVENTS_PER_STOCK,
            )
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    windows = alternative_window_summary(prices, market, baseline)

    common = prices.dropna(how="any")
    benchmark = equal_weight_buy_and_hold(common)
    portfolio_runs = {}
    metric_rows = [performance_metrics(benchmark, "Equal-weight focus basket")]
    for threshold in [0.05, 0.10, 0.15]:
        run = simulate_fixed_slot_strategy(
            prices,
            events,
            threshold=threshold,
            hold_days=63,
            position_weight=1.0 / len(FOCUS_TICKERS),
            max_positions=len(FOCUS_TICKERS),
            transaction_cost_bps=5.0,
        )
        portfolio_runs[threshold] = run
        daily = run["daily"]
        metric_rows.append(
            performance_metrics(
                daily["wealth"],
                f"Buy after {threshold:.0%} surge",
                exposure=daily["exposure"],
                position_count=daily["position_count"],
            )
        )
    portfolio_summary = pd.DataFrame(metric_rows)

    # Only aggregate/summary outputs are written. Price histories and event rows remain in memory.
    pooled.to_csv(OUTPUT_DIR / "surge_pooled_edges.csv", index=False)
    cooldown_summary.to_csv(OUTPUT_DIR / "surge_cooldown_summary.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "surge_bootstrap_intervals.csv", index=False)
    windows.to_csv(OUTPUT_DIR / "surge_window_sensitivity.csv", index=False)
    portfolio_summary.to_csv(OUTPUT_DIR / "surge_portfolio_summary.csv", index=False)

    save_edge_plot(pooled, FIGURE_DIR)
    save_portfolio_plot(benchmark, portfolio_runs, FIGURE_DIR)

    print("\nPrimary event-study summary")
    print(pooled.to_string(index=False))
    print("\n21-trading-day cooldown sensitivity")
    print(cooldown_summary.to_string(index=False))
    print("\nBootstrap intervals for the 10% / 3-month result")
    print(bootstrap.to_string(index=False))
    print("\nPortfolio summary")
    print(portfolio_summary.to_string(index=False))
    print("\nRaw price histories and event-level rows were not saved.")


if __name__ == "__main__":
    main()
