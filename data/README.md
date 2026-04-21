# Synthetic data generator

`datagen.py` simulates a 180-day ground-truth epidemic trajectory and a
negative-binomial-distributed ED-visit observation stream for posterior
recovery checks. All configuration is module-level; edit the script to
change it.

## Usage

From the repository root:

```
python data/datagen.py
```

Writes to `data/synthetic_180/`:

- `daily_infections.csv` — 180 rows: `date`, `true_infections`, `true_rt`
- `daily_ed_visits.csv` — 180 rows: `date`, `geo_value`, `disease`, `ed_visits`
- `true_parameters.json` — ground truth for posterior recovery

## R(t) schedule

Three 60-day piecewise-linear segments, deterministic (no noise):

| phase | days | start | end |
|---|---|---|---|
| 1 | 60 | 1.20 | 0.80 |
| 2 | 60 | 0.80 | 1.15 |
| 3 | 60 | 1.15 | 0.85 |

## Data-generating process

1. **Seed the warm-up** (`N_INIT = 50` days before day 0). Infections
   grow exponentially at the Lotka-Euler rate `r` implied by
   `R(0) = 1.2` and the generation interval PMF. `r` is found by
   `scipy.optimize.brentq` solving `R · Σ w_k · exp(−r(k+1)) = 1`.
   The seed is anchored so that the first observation day, produced
   by the first renewal step, equals
   `i0_total = I0_PER_CAPITA * POPULATION`.

2. **Run the renewal equation** forward for 180 days:
   ```
   I(t) = R(t) * sum_{k=1..S} I(t-k) * w(k-1)
   ```
   where `S = len(GEN_INT_PMF) = 7`. The loop is sequential; FFT
   cannot replace it because each `I(t)` depends on values just
   computed.

3. **Convolve infections with the ED delay PMF** via
   `scipy.signal.fftconvolve` and scale by `IEDR`:
   ```
   mu(t) = IEDR * sum_{d=0..D-1} I(t-d) * pi(d)
   ```
   Expected counts are floored at 1 before sampling.

4. **Sample observations** from NegativeBinomial2(mu, concentration).

5. **Slice into warm-up and observation.** The first `N_INIT = 50`
   entries of `I` are saved as metadata in `true_parameters.json` as
   `pre_observation_infections`; the Stan model consumes the last
   `max(D, S)` of these as its `J` input. The remaining 180 entries
   are the observation window.

## Constants

| name | value | meaning |
|---|---|---|
| `POPULATION` | 39,512,223 | denominator for per-capita seed |
| `I0_PER_CAPITA` | 5e-4 | seed anchor; `i0_total = I0_PER_CAPITA * POPULATION` is the infection count on the first observation day |
| `N_INIT` | 50 | warm-up days before day 0 |
| `START_DATE` | 2025-01-01 | date for day 0 of the observation window |
| `IEDR` | 0.0075 | infection-to-ED-visit ascertainment rate |
| `NEGBINOM_CONCENTRATION_ED` | 50.0 | NB2 concentration (higher ⇒ less overdispersion) |
| `RNG_SEED` | 20240101 | RNG seed |
| `GEN_INT_PMF` | length 7 | generation interval PMF
| `ED_DELAY_PMF` | length 12 | infection-to-ED-visit delay PMF |
