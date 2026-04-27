# Synthetic data generator

`datagen.py` simulates a ground-truth epidemic trajectory and one or
more negative-binomial-distributed observation streams for posterior
recovery checks.

## Usage

From the repository root:

```
python data/datagen.py 26
```

The positional argument is the requested number of weeks. It must be at
least 9, and is rounded up to the next multiple of 3 so the trajectory
can be split into three equal phases. For example, `26` weeks becomes
`27` weeks, or `189` days.

By default, output is written to `data/synthetic_<n_days>/`. You can
override this with `--output-dir`:

```
python data/datagen.py 26 --output-dir data/my_synthetic_run
```

Observation streams are configured in `data/observation_signals.json`
by default. Use `--signal-config` to provide another JSON file:

```
python data/datagen.py 26 --signal-config data/my_signals.json
```

Each run writes:

- `daily_infections.csv` — `n_days` rows: `date`, `true_infections`, `true_rt`
- `daily_<signal_name>.csv` — `n_days` rows: `date`, metadata columns, `observed_count`
- `true_parameters.json` — ground truth for posterior recovery

## Signal configuration

Signal configuration has shared metadata and one record per observation
signal:

```json
{
  "metadata": {
    "geo_value": "CA",
    "disease": "COVID-19"
  },
  "signals": [
    {
      "name": "ed_visits",
      "ascertainment_rate": 0.0075,
      "delay_pmf": [0.25, 0.5, 0.25],
      "negbinom_concentration": 50.0
    }
  ]
}
```

Signal names must be lowercase snake_case and filesystem-safe. Delay
PMFs must be nonnegative and sum to 1.0 within tolerance.

## R(t) schedule

Three equal-length piecewise-linear segments, deterministic (no noise):

| phase | days | start | end |
|---|---|---|---|
| 1 | `n_days / 3` | 1.20 | 0.80 |
| 2 | `n_days / 3` | 0.80 | 1.15 |
| 3 | `n_days / 3` | 1.15 | 0.85 |

## Data-generating process

1. **Seed the warm-up** (`n_init` days before day 0). Infections
   grow exponentially at the Lotka-Euler rate `r` implied by
   `R(0) = 1.2` and the generation interval PMF. `r` is found by
   `scipy.optimize.brentq` solving `R · Σ w_k · exp(−r(k+1)) = 1`.
   The seed is anchored so that the first observation day, produced
   by the first renewal step, equals
   `i0_total = I0_PER_CAPITA * POPULATION`.

2. **Run the renewal equation** forward for `n_days`:
   ```
   I(t) = R(t) * sum_{k=1..S} I(t-k) * w(k-1)
   ```
   where `S = len(GEN_INT_PMF) = 7`. The loop is sequential; FFT
   cannot replace it because each `I(t)` depends on values just
   computed.

3. **Convolve infections with each signal delay PMF** via
   `scipy.signal.fftconvolve` and scale by the signal ascertainment rate:
   ```
   mu(t) = ascertainment_rate * sum_{d=0..D-1} I(t-d) * pi(d)
   ```
   Expected counts are floored at 1 before sampling.

4. **Sample observations** from NegativeBinomial2(mu, concentration)
   independently for each configured signal.

5. **Slice into warm-up and observation.** The first `n_init`
   entries of `I` are saved as metadata in `true_parameters.json` as
   `pre_observation_infections`; the Stan model consumes the last
   `max(D, S)` of these as its `J` input. The remaining `n_days` entries
   are the observation window.

## Constants

| name | value | meaning |
|---|---|---|
| `POPULATION` | 39,512,223 | denominator for per-capita seed |
| `I0_PER_CAPITA` | 5e-4 | seed anchor; `i0_total = I0_PER_CAPITA * POPULATION` is the infection count on the first observation day |
| `MIN_N_INIT` | 50 | minimum warm-up days before day 0 |
| `START_DATE` | 2025-01-01 | date for day 0 of the observation window |
| `RNG_SEED` | 20240101 | RNG seed |
| `GEN_INT_PMF` | length 7 | generation interval PMF
