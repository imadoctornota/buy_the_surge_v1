"""Small plotting helpers for the surge study."""

from pathlib import Path

import matplotlib.pyplot as plt


def save_edge_plot(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    subset = summary.query("outcome == 'relative' and horizon == '3m'").copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar((100 * subset["threshold"]).astype(int).astype(str) + "%", 100 * subset["equal_stock_edge"])
    ax.axhline(0, linewidth=1)
    ax.set(
        xlabel="One-day surge threshold",
        ylabel="3-month SPY-relative edge (percentage points)",
        title="Forward returns after one-day surges",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "surge_relative_edge.png", dpi=180)
    plt.close(fig)


def save_portfolio_plot(benchmark, portfolio_runs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(benchmark.index, benchmark, label="Equal-weight buy and hold")
    for threshold, run in portfolio_runs.items():
        ax.plot(run["daily"].index, run["daily"]["wealth"], label=f"Buy after {threshold:.0%} surge")
    ax.set_yscale("log")
    ax.set(xlabel="Date", ylabel="Growth of $1", title="Surge-signal portfolios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "surge_portfolios.png", dpi=180)
    plt.close(fig)
