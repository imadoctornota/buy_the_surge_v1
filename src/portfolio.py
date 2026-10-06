"""Portfolio simulation used as a separate test of signal usefulness."""

import numpy as np
import pandas as pd


def equal_weight_buy_and_hold(prices):
    common = prices.dropna(how="any")
    weights = pd.Series(1.0 / common.shape[1], index=common.columns)
    shares = weights / common.iloc[0]
    wealth = common.mul(shares, axis=1).sum(axis=1)
    return wealth / wealth.iloc[0]


def simulate_fixed_slot_strategy(
    prices,
    events,
    threshold,
    hold_days=63,
    position_weight=None,
    max_positions=None,
    transaction_cost_bps=5.0,
):
    common = prices.dropna(how="any").copy()
    tickers = list(common.columns)
    if position_weight is None:
        position_weight = 1.0 / len(tickers)
    if max_positions is None:
        max_positions = len(tickers)

    index = common.index
    index_position = pd.Series(np.arange(len(index)), index=index)

    entry_map = {}
    selected = events.loc[events["threshold"].eq(threshold)].sort_values("signal_date")
    for row in selected.itertuples(index=False):
        if row.signal_date not in index_position.index:
            continue
        signal_i = int(index_position.loc[row.signal_date])
        entry_i = signal_i + 1
        if entry_i >= len(index):
            continue
        entry_map.setdefault(index[entry_i], []).append(row.ticker)

    cash = 1.0
    positions = {}
    records = []
    trade_rows = []
    cost_rate = transaction_cost_bps / 10000.0

    for i, date in enumerate(index):
        px = common.loc[date]

        # Close positions whose holding period has ended.
        for ticker in list(positions):
            position = positions[ticker]
            if i >= position["exit_i"]:
                gross = position["shares"] * px[ticker]
                fee = gross * cost_rate
                cash += gross - fee
                trade_rows.append({"ticker": ticker, "date": date, "side": "exit", "value": gross, "fee": fee})
                del positions[ticker]

        wealth_before_entries = cash + sum(pos["shares"] * px[ticker] for ticker, pos in positions.items())

        # Enter at today's close for signals observed at the previous close.
        for ticker in entry_map.get(date, []):
            if ticker in positions or len(positions) >= max_positions:
                continue
            target_value = wealth_before_entries * position_weight
            fee = target_value * cost_rate
            required_cash = target_value + fee
            if required_cash > cash:
                continue
            shares = target_value / px[ticker]
            cash -= required_cash
            positions[ticker] = {"shares": shares, "exit_i": min(i + hold_days, len(index) - 1)}
            trade_rows.append({"ticker": ticker, "date": date, "side": "entry", "value": target_value, "fee": fee})

        invested = sum(pos["shares"] * px[ticker] for ticker, pos in positions.items())
        wealth = cash + invested
        records.append(
            {
                "date": date,
                "wealth": wealth,
                "exposure": invested / wealth if wealth > 0 else 0.0,
                "position_count": len(positions),
            }
        )

    daily = pd.DataFrame(records).set_index("date")
    daily["returns"] = daily["wealth"].pct_change().fillna(0.0)
    return {"daily": daily, "trades": pd.DataFrame(trade_rows)}


def performance_metrics(wealth, label, exposure=None, position_count=None):
    wealth = wealth.dropna()
    returns = wealth.pct_change().dropna()
    years = (wealth.index[-1] - wealth.index[0]).days / 365.25
    drawdown = wealth / wealth.cummax() - 1.0

    row = {
        "strategy": label,
        "start": wealth.index[0],
        "end": wealth.index[-1],
        "total_return": wealth.iloc[-1] / wealth.iloc[0] - 1.0,
        "cagr": (wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / years) - 1.0,
        "annualized_vol": returns.std() * np.sqrt(252),
        "sharpe_0rf": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan,
        "max_drawdown": drawdown.min(),
    }
    if exposure is not None:
        row["avg_invested_fraction"] = exposure.reindex(wealth.index).mean()
    if position_count is not None:
        row["avg_positions"] = position_count.reindex(wealth.index).mean()
    return row
