"""Plot observed infections and ED visits from the synthetic_180 dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "synthetic_180"


def load_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the daily infections and ED visits frames as Polars DataFrames."""
    infections = pl.read_csv(DATA_DIR / "daily_infections.csv").with_columns(
        pl.col("date").str.to_date()
    )
    ed_visits = pl.read_csv(DATA_DIR / "daily_ed_visits.csv").with_columns(
        pl.col("date").str.to_date()
    )
    return infections, ed_visits


def plot(infections: pl.DataFrame, ed_visits: pl.DataFrame, out_path: Path) -> None:
    """Render a two-panel plot of true infections and observed ED visits."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(
        infections["date"].to_list(),
        infections["true_infections"].to_list(),
        color="darkblue",
    )
    axes[0].set_ylabel("True infections")
    axes[0].set_title("Synthetic dataset (180 days)")
    axes[1].plot(
        ed_visits["date"].to_list(),
        ed_visits["ed_visits"].to_list(),
        color="firebrick",
        marker="o",
        markersize=2,
        linewidth=0.8,
    )
    axes[1].set_ylabel("Observed ED visits")
    axes[1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Load the synthetic data and save the two-panel plot."""
    infections, ed_visits = load_frames()
    out_path = DATA_DIR / "synthetic_180_overview.png"
    plot(infections, ed_visits, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
