# numpydoc ignore=ES01,SA01,EX01
"""
Generate synthetic observation data from a known R(t).

This script defines R(t) directly, runs a discrete renewal equation
forward, and convolves with each configured observation delay PMF to
produce NegativeBinomial-distributed observation streams. All true
parameters are saved alongside the synthetic observations for posterior
recovery checks.

Outputs (under ``synthetic_<n_days>`` by default):

- ``true_parameters.json``
- ``daily_infections.csv``
- ``daily_<signal_name>.csv``
"""

from __future__ import annotations

import json
import math
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.optimize import brentq
from scipy.signal import fftconvolve

ROOT: Path = Path(__file__).resolve().parent
DEFAULT_SIGNAL_CONFIG: Path = ROOT / "observation_signals.json"

RNG_SEED: int = 20240101
POPULATION: int = 39_512_223

GEN_INT_PMF: npt.NDArray[np.float64] = np.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
)

I0_PER_CAPITA: float = 5e-4

START_DATE: date = date(2025, 1, 1)
MIN_N_INIT: int = 50
MIN_WEEKS: int = 9
N_RT_PHASES: int = 3
SIGNAL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
PMF_SUM_TOLERANCE: float = 1e-6

RT_PHASE_ENDPOINTS: tuple[tuple[float, float], ...] = (
    (1.2, 0.8),
    (0.8, 1.15),
    (1.15, 0.85),
)


@dataclass(frozen=True)
class SignalConfig:
    """Observation-process configuration for one signal."""

    name: str
    ascertainment_rate: float
    delay_pmf: npt.NDArray[np.float64]
    negbinom_concentration: float

    @property
    def output_filename(self) -> str:
        """Return the CSV filename generated for this signal."""
        return f"daily_{self.name}.csv"


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Generate synthetic renewal-process data for a requested number "
            "of weeks. Weeks are rounded up to a multiple of 3."
        )
    )
    parser.add_argument(
        "weeks",
        type=int,
        help=f"number of weeks to generate; minimum {MIN_WEEKS}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "directory for generated files; defaults to "
            f"{ROOT.name}/synthetic_<n_days>"
        ),
    )
    parser.add_argument(
        "-s",
        "--signal-config",
        type=Path,
        default=DEFAULT_SIGNAL_CONFIG,
        help=(
            "JSON file with observation signal metadata and parameters; "
            f"defaults to {DEFAULT_SIGNAL_CONFIG.relative_to(ROOT)}"
        ),
    )
    return parser.parse_args()


def load_signal_config(path: Path) -> tuple[dict[str, Any], list[SignalConfig]]:
    """
    Load and validate observation signal configuration.

    Parameters
    ----------
    path : pathlib.Path
        JSON file containing top-level ``metadata`` and ``signals`` keys.

    Returns
    -------
    tuple[dict[str, Any], list[SignalConfig]]
        Shared metadata and validated per-signal configurations.
    """
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"could not find signal config file {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse signal config file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"signal config file {path} must be a JSON object")

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("signal config metadata must be an object")

    raw_signals = raw.get("signals")
    if not isinstance(raw_signals, list) or not raw_signals:
        raise ValueError("signal config must contain a non-empty signals array")

    signals: list[SignalConfig] = []
    seen_names: set[str] = set()
    for i, raw_signal in enumerate(raw_signals, start=1):
        if not isinstance(raw_signal, dict):
            raise ValueError(f"signal {i} must be an object")

        name = raw_signal.get("name")
        if not isinstance(name, str) or not SIGNAL_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"signal {i} name must be lowercase snake_case starting with a letter"
            )
        if name in seen_names:
            raise ValueError(f"duplicate signal name: {name}")
        seen_names.add(name)

        ascertainment_rate = raw_signal.get("ascertainment_rate")
        if type(ascertainment_rate) not in (int, float) or ascertainment_rate <= 0:
            raise ValueError(f"{name}: ascertainment_rate must be positive")

        concentration = raw_signal.get("negbinom_concentration")
        if type(concentration) not in (int, float) or concentration <= 0:
            raise ValueError(f"{name}: negbinom_concentration must be positive")

        delay_pmf_raw = raw_signal.get("delay_pmf")
        if not isinstance(delay_pmf_raw, list) or not delay_pmf_raw:
            raise ValueError(f"{name}: delay_pmf must be a non-empty array")
        try:
            delay_pmf = np.array(delay_pmf_raw, dtype=float)
        except ValueError as exc:
            raise ValueError(f"{name}: delay_pmf must contain only numbers") from exc
        if not np.all(np.isfinite(delay_pmf)):
            raise ValueError(f"{name}: delay_pmf must contain only finite values")
        if np.any(delay_pmf < 0):
            raise ValueError(f"{name}: delay_pmf must be nonnegative")
        if not math.isclose(float(np.sum(delay_pmf)), 1.0, abs_tol=PMF_SUM_TOLERANCE):
            raise ValueError(
                f"{name}: delay_pmf must sum to 1.0 within {PMF_SUM_TOLERANCE}"
            )

        signals.append(
            SignalConfig(
                name=name,
                ascertainment_rate=float(ascertainment_rate),
                delay_pmf=delay_pmf,
                negbinom_concentration=float(concentration),
            )
        )

    return metadata, signals


def rounded_weeks(weeks: int) -> int:
    """
    Round a requested week count up to the supported trajectory length.

    Parameters
    ----------
    weeks : int
        Requested number of weeks.

    Returns
    -------
    int
        Week count after enforcing the minimum and rounding up to a
        multiple of ``N_RT_PHASES``.
    """
    if weeks < MIN_WEEKS:
        raise ValueError(f"weeks must be at least {MIN_WEEKS}; got {weeks}")
    return int(math.ceil(weeks / N_RT_PHASES) * N_RT_PHASES)


def default_output_dir(n_days: int) -> Path:
    """
    Return the default output directory for a generated dataset.

    Parameters
    ----------
    n_days : int
        Number of observation days in the generated dataset.

    Returns
    -------
    pathlib.Path
        Default directory named according to the number of generated days.
    """
    return ROOT / f"synthetic_{n_days}"


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


def build_true_rt(n_days: int) -> npt.NDArray[np.float64]:
    """
    Build a piecewise-linear true R(t) trajectory.

    Phases: decline from 1.2 to 0.8, rise from 0.8 to 1.15,
    decline from 1.15 to 0.85. Each phase has equal length.

    Parameters
    ----------
    n_days : int
        Number of days in the trajectory. Must be divisible by 3.

    Returns
    -------
    np.ndarray
        R(t) trajectory of length ``n_days``.
    """
    if n_days % N_RT_PHASES != 0:
        raise ValueError(f"n_days must be divisible by {N_RT_PHASES}; got {n_days}")
    phase_days = n_days // N_RT_PHASES
    return np.concatenate(
        [
            np.linspace(start, end, phase_days, endpoint=False)
            for start, end in RT_PHASE_ENDPOINTS
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


def generate(
    weeks: int,
    output_dir: Path | None = None,
    signal_config_path: Path = DEFAULT_SIGNAL_CONFIG,
) -> None:
    """Generate all synthetic data files and the true-parameter JSON."""
    signal_metadata, signals = load_signal_config(signal_config_path)
    n_weeks = rounded_weeks(weeks)
    n_days = n_weeks * 7
    n_init = max(
        MIN_N_INIT,
        len(GEN_INT_PMF),
        *(len(signal.delay_pmf) for signal in signals),
    )
    output_dir = output_dir or default_output_dir(n_days)

    rng = np.random.default_rng(RNG_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_rt = build_true_rt(n_days)
    phase_days = n_days // N_RT_PHASES
    i0_total = I0_PER_CAPITA * POPULATION

    infections_full = run_renewal(true_rt, GEN_INT_PMF, i0_total, n_init)
    infections_obs = infections_full[n_init:]

    obs_dates = [START_DATE + timedelta(days=i) for i in range(n_days)]

    infections_df = pl.DataFrame(
        {
            "date": obs_dates,
            "true_infections": infections_obs.tolist(),
            "true_rt": true_rt.tolist(),
        }
    )
    infections_df.write_csv(output_dir / "daily_infections.csv")

    signal_outputs: dict[str, str] = {}
    signal_means: dict[str, float] = {}
    for signal in signals:
        expected_full = (
            convolve_with_pmf(infections_full, signal.delay_pmf)
            * signal.ascertainment_rate
        )
        expected = np.maximum(expected_full[n_init:], 1.0)
        observed = sample_negbinom(expected, signal.negbinom_concentration, rng)

        signal_df = pl.DataFrame(
            {
                "date": obs_dates,
                **{key: [value] * n_days for key, value in signal_metadata.items()},
                "observed_count": observed.tolist(),
            }
        )
        signal_df.write_csv(output_dir / signal.output_filename)
        signal_outputs[signal.name] = signal.output_filename
        signal_means[signal.name] = float(np.mean(observed))

    true_params: dict[str, Any] = {
        "description": (
            "True parameters used to generate synthetic data. "
            "All values are known ground truth for posterior recovery checks."
        ),
        "population": POPULATION,
        "start_date": str(START_DATE),
        "requested_weeks": weeks,
        "n_weeks": n_weeks,
        "n_days": n_days,
        "n_init": n_init,
        "rng_seed": RNG_SEED,
        "generation_interval_pmf": GEN_INT_PMF.tolist(),
        "i0_per_capita": I0_PER_CAPITA,
        "rt_trajectory": {
            f"phase_{i}": {"days": phase_days, "start": start, "end": end}
            for i, (start, end) in enumerate(RT_PHASE_ENDPOINTS, start=1)
        },
        "signal_config_file": str(signal_config_path),
        "signal_outputs": signal_outputs,
        "pre_observation_infections": infections_full[:n_init].tolist(),
        "true_rt": true_rt.tolist(),
    }
    with open(output_dir / "true_parameters.json", "w") as f:
        json.dump(true_params, f, indent=2)
        f.write("\n")

    if n_weeks != weeks:
        print(f"Rounded requested weeks from {weeks} to {n_weeks}")
    print(f"Wrote {n_days} daily infection rows")
    for signal_name, mean_observed in signal_means.items():
        print(
            f"Wrote {n_days} daily {signal_name} rows "
            f"(mean {mean_observed:.0f}/day)"
        )
    print(f"Output directory: {output_dir}")


def main() -> None:
    """Run the command-line interface."""
    args = parse_args()
    try:
        generate(args.weeks, args.output_dir, args.signal_config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
