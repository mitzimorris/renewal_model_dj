# numpydoc ignore=ES01,SA01,EX01
"""
Generate 180-day synthetic ED visit data from a known R(t).

This script defines R(t) directly, runs a discrete renewal equation
forward, and convolves with the ED-visit delay PMF to produce a
NegativeBinomial-distributed observation stream. All true parameters
are saved alongside the synthetic observations for posterior recovery
checks.

Outputs (under ``OUTPUT_DIR``):

- ``true_parameters.json``
- ``daily_infections.csv``
- ``daily_ed_visits.csv``
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.optimize import brentq
from scipy.signal import fftconvolve

ROOT: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = ROOT / "synthetic_180"

RNG_SEED: int = 20240101
POPULATION: int = 39_512_223

GEN_INT_PMF: npt.NDArray[np.float64] = np.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
)

ED_DELAY_PMF: npt.NDArray[np.float64] = np.array(
    [
        0.0,
        0.0213253,
        0.17156943,
        0.23836233,
        0.20200046,
        0.14144434,
        0.09118459,
        0.0567108,
        0.03480426,
        0.0213253,
        0.01312726,
        0.00814594,
    ]
)

IEDR: float = 0.0075
I0_PER_CAPITA: float = 5e-4
NEGBINOM_CONCENTRATION_ED: float = 50.0

START_DATE: date = date(2025, 1, 1)
N_INIT: int = max(50, len(GEN_INT_PMF), len(ED_DELAY_PMF))


def r_approx_from_R(
    rt0: float,
    gen_int: npt.NDArray[np.float64],
) -> float:
    """
    Solve the Lotka-Euler equation for the exponential growth rate.

    Finds ``r`` such that ``R * sum_k w_k * exp(-r * (k + 1)) = 1``,
    which is the geometric-growth consistency condition for the
    discrete renewal equation with generation-interval PMF ``w``.

    Parameters
    ----------
    rt0 : float
        Reproduction number at the start of the trajectory.
    gen_int : np.ndarray
        Generation interval PMF, with ``gen_int[k]`` the mass at lag
        ``k + 1``.

    Returns
    -------
    float
        Intrinsic growth rate ``r``.
    """
    if abs(rt0 - 1.0) < 1e-12:
        return 0.0
    tau = np.arange(1, len(gen_int) + 1)

    def residual(r: float) -> float:
        """Evaluate the Lotka-Euler residual at growth rate ``r``."""
        return rt0 * float(np.sum(gen_int * np.exp(-r * tau))) - 1.0

    return float(brentq(residual, -2.0, 2.0))


def build_true_rt() -> npt.NDArray[np.float64]:
    """
    Build a piecewise-linear true R(t) trajectory.

    Phases: decline from 1.2 to 0.8 (60 d), rise from 0.8 to 1.15
    (60 d), decline from 1.15 to 0.85 (60 d).

    Returns
    -------
    np.ndarray
        R(t) trajectory of length 180.
    """
    segments = [
        (60, 1.2, 0.8),
        (60, 0.8, 1.15),
        (60, 1.15, 0.85),
    ]
    return np.concatenate(
        [
            np.linspace(start, end, length, endpoint=False)
            for length, start, end in segments
        ]
    )


def run_renewal(
    rt: npt.NDArray[np.float64],
    gen_int: npt.NDArray[np.float64],
    i0_total: float,
    n_init: int,
) -> npt.NDArray[np.float64]:
    """
    Run a discrete renewal equation forward in time.

    Seed infections are placed as an exponentially growing trajectory
    over ``n_init`` days using the Lotka-Euler growth rate implied by
    ``rt[0]``, then the renewal equation is applied for ``len(rt)``
    days. The renewal step is recursive and cannot be replaced by a
    single FFT convolution.

    Parameters
    ----------
    rt : np.ndarray
        Effective reproduction number over the observation window.
    gen_int : np.ndarray
        Generation interval PMF (sums to 1).
    i0_total : float
        Infections on the last day of the seed period.
    n_init : int
        Number of seed days before day 0 of the observation window.

    Returns
    -------
    np.ndarray
        Infections of shape ``(n_init + len(rt),)``.
    """
    n_days = len(rt)
    n_total = n_init + n_days
    infections = np.zeros(n_total)

    r0_approx = r_approx_from_R(float(rt[0]), gen_int)
    seed_times = np.arange(-n_init, 0)
    infections[:n_init] = i0_total * np.exp(r0_approx * seed_times)

    g_len = len(gen_int)
    for t in range(n_init, n_total):
        lookback = min(t, g_len)
        convolution = float(
            np.sum(infections[t - lookback : t][::-1] * gen_int[:lookback])
        )
        infections[t] = rt[t - n_init] * convolution

    return infections


def convolve_with_pmf(
    signal: npt.NDArray[np.float64],
    pmf: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Convolve a signal with a delay PMF using FFT, preserving length.

    Parameters
    ----------
    signal : np.ndarray
        Input time series.
    pmf : np.ndarray
        Delay PMF; ``pmf[d]`` is the mass at lag ``d``.

    Returns
    -------
    np.ndarray
        Convolved signal truncated to ``len(signal)``.
    """
    return fftconvolve(signal, pmf, mode="full")[: len(signal)]


def sample_negbinom(
    mu: npt.NDArray[np.float64],
    concentration: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """
    Sample from the NegativeBinomial2 (mean, concentration) parameterization.

    Parameters
    ----------
    mu : np.ndarray
        Mean values (must be positive).
    concentration : float
        Concentration parameter; higher means less overdispersion.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Integer counts.
    """
    mu = np.maximum(mu, 1e-10)
    p = concentration / (concentration + mu)
    return rng.negative_binomial(n=concentration, p=p)


def generate() -> None:
    """Generate all synthetic data files and the true-parameter JSON."""
    rng = np.random.default_rng(RNG_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    true_rt = build_true_rt()
    n_days = len(true_rt)
    i0_total = I0_PER_CAPITA * POPULATION

    infections_full = run_renewal(true_rt, GEN_INT_PMF, i0_total, N_INIT)
    infections_obs = infections_full[N_INIT:]

    obs_dates = [START_DATE + timedelta(days=i) for i in range(n_days)]

    expected_ed_full = convolve_with_pmf(infections_full, ED_DELAY_PMF) * IEDR
    expected_ed = np.maximum(expected_ed_full[N_INIT:], 1.0)
    ed_obs = sample_negbinom(expected_ed, NEGBINOM_CONCENTRATION_ED, rng)

    infections_df = pl.DataFrame(
        {
            "date": obs_dates,
            "true_infections": infections_obs.tolist(),
            "true_rt": true_rt.tolist(),
        }
    )
    infections_df.write_csv(OUTPUT_DIR / "daily_infections.csv")

    ed_df = pl.DataFrame(
        {
            "date": obs_dates,
            "geo_value": ["CA"] * n_days,
            "disease": ["COVID-19"] * n_days,
            "ed_visits": ed_obs.tolist(),
        }
    )
    ed_df.write_csv(OUTPUT_DIR / "daily_ed_visits.csv")

    true_params: dict[str, Any] = {
        "description": (
            "True parameters used to generate synthetic data. "
            "All values are known ground truth for posterior recovery checks."
        ),
        "population": POPULATION,
        "start_date": str(START_DATE),
        "n_days": n_days,
        "n_init": N_INIT,
        "rng_seed": RNG_SEED,
        "generation_interval_pmf": GEN_INT_PMF.tolist(),
        "i0_per_capita": I0_PER_CAPITA,
        "rt_trajectory": {
            "phase_1": {"days": 60, "start": 1.2, "end": 0.8},
            "phase_2": {"days": 60, "start": 0.8, "end": 1.15},
            "phase_3": {"days": 60, "start": 1.15, "end": 0.85},
        },
        "ed_visits": {
            "iedr": IEDR,
            "delay_pmf": ED_DELAY_PMF.tolist(),
            "negbinom_concentration": NEGBINOM_CONCENTRATION_ED,
            "temporal_resolution": "daily",
        },
        "pre_observation_infections": infections_full[:N_INIT].tolist(),
        "true_rt": true_rt.tolist(),
    }
    with open(OUTPUT_DIR / "true_parameters.json", "w") as f:
        json.dump(true_params, f, indent=2)
        f.write("\n")

    print(f"Wrote {n_days} daily infection rows")
    print(f"Wrote {n_days} daily ED visit rows (mean {float(np.mean(ed_obs)):.0f}/day)")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate()
