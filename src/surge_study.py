"""Core event-study functions for the stock-surge analysis."""

import numpy as np
import pandas as pd


def detect_surge_events(surge_returns, thresholds, window=1):
    rows = []
    for ticker in surge_returns.columns:
        series = surge_returns[ticker].dropna()
        for threshold in thresholds:
            crossed = (series >= threshold) & (series.shift(1) < threshold)
            for date in series.index[crossed.fillna(False)]:
                rows.append(
                    {
                        "ticker": ticker,
                        "signal_date": date,
                        "threshold": threshold,
                        "actual_surge": float(series.loc[date]),
                        "surge_window": window,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["ticker", "signal_date", "threshold", "actual_surge", "surge_window"])
    return pd.DataFrame(rows).sort_values(["ticker", "threshold", "signal_date"]).reset_index(drop=True)


def apply_cooldown(events, prices, cooldown_days=21):
    kept = []
    for (ticker, threshold), group in events.groupby(["ticker", "threshold"], sort=False):
        trading_index = prices[ticker].dropna().index
        positions = pd.Series(np.arange(len(trading_index)), index=trading_index)
        last_position = None
        for row in group.sort_values("signal_date").itertuples(index=False):
            if row.signal_date not in positions.index:
                continue
            position = int(positions.loc[row.signal_date])
            if last_position is None or position - last_position >= cooldown_days:
                kept.append(row._asdict())
                last_position = position
    return pd.DataFrame(kept)


def add_forward_returns(events, prices, market, horizons, entry_lag=1):
    rows = []
    for event in events.itertuples(index=False):
        aligned = pd.concat(
            [prices[event.ticker].rename("stock"), market.rename("market")], axis=1
        ).dropna()
        if event.signal_date not in aligned.index:
            continue

        signal_i = aligned.index.get_loc(event.signal_date)
        entry_i = signal_i + entry_lag
        if entry_i >= len(aligned):
            continue

        row = event._asdict()
        row["entry_date"] = aligned.index[entry_i]
        for label, horizon in horizons.items():
            exit_i = entry_i + horizon
            if exit_i >= len(aligned):
                row[f"return_{label}"] = np.nan
                row[f"market_return_{label}"] = np.nan
                row[f"relative_return_{label}"] = np.nan
                continue

            stock_return = float(aligned["stock"].iloc[exit_i] / aligned["stock"].iloc[entry_i] - 1.0)
            market_return = float(aligned["market"].iloc[exit_i] / aligned["market"].iloc[entry_i] - 1.0)
            row[f"return_{label}"] = stock_return
            row[f"market_return_{label}"] = market_return
            row[f"relative_return_{label}"] = (1.0 + stock_return) / (1.0 + market_return) - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def ordinary_day_baselines(prices, market, horizons, entry_lag=1):
    rows = []
    for ticker in prices.columns:
        aligned = pd.concat([prices[ticker].rename("stock"), market.rename("market")], axis=1).dropna()
        for label, horizon in horizons.items():
            stock_forward = aligned["stock"].shift(-(entry_lag + horizon)) / aligned["stock"].shift(-entry_lag) - 1.0
            market_forward = aligned["market"].shift(-(entry_lag + horizon)) / aligned["market"].shift(-entry_lag) - 1.0
            relative = (1.0 + stock_forward) / (1.0 + market_forward) - 1.0
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": label,
                    "ordinary_raw_mean": float(stock_forward.mean()),
                    "ordinary_raw_median": float(stock_forward.median()),
                    "ordinary_relative_mean": float(relative.mean()),
                    "ordinary_relative_median": float(relative.median()),
                }
            )
    return pd.DataFrame(rows)


def pooled_edge_summary(results, baseline, thresholds, horizons, min_events_per_stock=3):
    rows = []
    for threshold in thresholds:
        threshold_results = results.loc[results["threshold"].eq(threshold)]
        for label in horizons:
            base = baseline.loc[baseline["horizon"].eq(label)]
            for outcome, prefix in [("raw", "return"), ("relative", "relative_return")]:
                value_col = f"{prefix}_{label}"
                if value_col not in threshold_results:
                    continue
                per_stock = (
                    threshold_results[["ticker", value_col]]
                    .dropna()
                    .groupby("ticker")[value_col]
                    .agg(["mean", "median", "count"])
                    .reset_index()
                )
                per_stock = per_stock.loc[per_stock["count"] >= min_events_per_stock].copy()
                if per_stock.empty:
                    continue

                baseline_col = f"ordinary_{outcome}_mean"
                baseline_median_col = f"ordinary_{outcome}_median"
                per_stock = per_stock.merge(
                    base[["ticker", baseline_col, baseline_median_col]], on="ticker", how="left"
                )
                rows.append(
                    {
                        "outcome": outcome,
                        "threshold": threshold,
                        "horizon": label,
                        "n_events": int(per_stock["count"].sum()),
                        "n_stocks": int(len(per_stock)),
                        "equal_stock_post_mean": float(per_stock["mean"].mean()),
                        "equal_stock_baseline_mean": float(per_stock[baseline_col].mean()),
                        "equal_stock_edge": float((per_stock["mean"] - per_stock[baseline_col]).mean()),
                        "equal_stock_post_median": float(per_stock["median"].mean()),
                        "equal_stock_median_edge": float(
                            (per_stock["median"] - per_stock[baseline_median_col]).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def hierarchical_bootstrap(
    results,
    baseline,
    threshold,
    horizon,
    outcome="raw",
    draws=2000,
    seed=42,
    min_events_per_stock=3,
):
    value_col = f"return_{horizon}" if outcome == "raw" else f"relative_return_{horizon}"
    baseline_col = f"ordinary_{outcome}_mean"

    subset = results.loc[results["threshold"].eq(threshold), ["ticker", value_col]].dropna()
    counts = subset.groupby("ticker").size()
    eligible = counts.index[counts >= min_events_per_stock].tolist()
    if not eligible:
        return {"outcome": outcome, "threshold": threshold, "horizon": horizon, "draws": 0,
                "edge_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    baseline_map = baseline.loc[baseline["horizon"].eq(horizon)].set_index("ticker")[baseline_col]
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)

    for i in range(draws):
        sampled_stocks = rng.choice(eligible, size=len(eligible), replace=True)
        stock_edges = []
        for ticker in sampled_stocks:
            values = subset.loc[subset["ticker"].eq(ticker), value_col].to_numpy()
            resampled = rng.choice(values, size=len(values), replace=True)
            stock_edges.append(float(resampled.mean() - baseline_map.loc[ticker]))
        boot[i] = np.mean(stock_edges)

    return {
        "outcome": outcome,
        "threshold": threshold,
        "horizon": horizon,
        "draws": draws,
        "edge_mean": float(boot.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }
