import jax
import jax.numpy as jnp
import jax.random as jrd
from jax.scipy import stats
from jax.typing import ArrayLike
import blackjax

import json
from pathlib import Path
import polars as pl
from typing import Any

from util import positive, real
from util_renewal import solve_initial_growth_rate

# These are the primary exports of this module:
__all__ = [
    "log_density",
    "log_density_vec",
    "constraints",
    "generated_quantities",
    "generated_quantities_vec",
]


# data and transformed data
# closed over - specify directly here

with open(Path("data/synth_hosp_ed_189/daily_ed_visits.csv")) as f:
    observations = pl.read_csv(f)
y = jnp.array(observations["observed_count"].to_numpy())

w: jax.Array = jnp.array(
    [0.6326975, 0.2327564, 0.0856263, 0.03150015, 0.01158826, 0.00426308, 0.0015683]
)
pi: jax.Array = jnp.array(
    [0.0, 0.0213253, 0.17156943, 0.23836233, 0.20200046, 0.14144434, 0.09118459,
         0.0567108, 0.03480426, 0.0213253, 0.01312726, 0.00814594]
)

T = len(y)
S = len(w)
D = len(pi)


# Stan simplex data type constraints - OK to do on data - cannot be done in jit'd code
def validate_simplex(x, tol=1e-6):
    assert bool(jnp.all(x >= 0))
    assert abs(float(jnp.sum(x)) - 1.0) < tol

validate_simplex(w)
validate_simplex(pi)

rt0: jnp.float64 = 1.2
i0_scale: jnp.float64 = 0.0005
population_size: jnp.float64 = 39512223

L = max(len(w), len(pi))
w_rev = jnp.flip(w)
pi_rev = jnp.flip(pi)

r0_approx = solve_initial_growth_rate(rt0, w)
pre_observation_infections = (
    population_size * i0_scale * jnp.exp(r0_approx * jnp.arange(-L, 0))
)


## defining everything using jax.scipy densities
def log_posterior(params):
    # construct vectors mu and I iteratively
    rt = jnp.exp(params["log_r"])
    def renewal_step(ts, k):
        history_I = jax.lax.dynamic_slice(ts, (k - S,), (S,))
        I_k = rt[k - L] * jnp.dot(history_I, w_rev)
        ts = ts.at[k].set(I_k)

        history_mu = jax.lax.dynamic_slice(ts, (k - D + 1,), (D,))
        mu_t = params["alpha"] * jnp.dot(history_mu, pi_rev)

        return ts, (I_k, mu_t)

    I0 = jnp.concatenate([pre_observation_infections, jnp.zeros(T)])
    ks = jnp.arange(L, L + T)
    final_ts, (I, mu) = jax.lax.scan(renewal_step, I0, ks)

    lp = 0.0

    ## likelihood   y ~ neg_binomial_2(mu, phi);
    ## we need to parameterize negative binomial in terms of mean, concentration
    phi = jnp.reciprocal(jnp.power(params["inv_sqrt_phi"], 2))
    n = phi
    p = phi / (phi + mu)
    lp += jnp.sum(stats.nbinom.logpmf(y, n=n, p=p))

    ## priors
    lp += jnp.sum(stats.beta.logpdf(params["alpha"], a=1, b=100, loc=0.0, scale=1.0))
    lp += jnp.sum(stats.norm.logpdf(params["inv_sqrt_phi"], loc=0.0, scale=1.0))
    lp += jnp.sum(stats.norm.logpdf(params["sigma_rw"], loc=0.0, scale=0.5))

    ## random walk prior on log_r
    lp += jnp.sum(stats.norm.logpdf(params["log_r"][0], loc=0.0, scale=0.5))
    lp += jnp.sum(stats.norm.logpdf(params["log_r"][1:], loc=params["log_r"][0:-1], scale=params["sigma_rw"]))

    return lp

    
## need to apply transforms
def transform(params):
    t_log_r = jnp.array(params["log_r"])
    t_alpha = jnp.log(params["alpha"])
    t_inv_sqrt_phi = jnp.log(params["inv_sqrt_phi"])
    t_sigma_rw = jnp.log(params["sigma_rw"])
    t_params = {
        "log_r": t_log_r,
        "alpha": t_alpha,
        "inv_sqrt_phi": t_inv_sqrt_phi,
        "sigma_rw": t_sigma_rw
    }
    return t_params

def inv_transform(t_params):
    log_adjust = 0.0
    log_r = jnp.array(t_params["log_r"])
    alpha = jnp.exp(t_params["alpha"])
    log_adjust += t_params["alpha"]
    inv_sqrt_phi = jnp.exp(t_params["inv_sqrt_phi"])
    log_adjust += t_params["inv_sqrt_phi"]
    sigma_rw = jnp.exp(t_params["sigma_rw"])
    log_adjust += t_params["sigma_rw"]
    params = {
        "log_r": log_r,
        "alpha": alpha,
        "inv_sqrt_phi": inv_sqrt_phi,
        "sigma_rw": sigma_rw
    }
    return params, log_adjust

def log_posterior_transformed(t_params):
    params, log_adjust = inv_transform(t_params)
    log_post = log_posterior(params)
    return log_adjust + log_post

def random_init_transformed(key):
    key0, key1, key2, key3 = jrd.split(key, 4)
    t_log_r = jrd.normal(key0, shape=(y.shape[0]))
    t_alpha = jrd.normal(key1)
    t_inv_sqrt_phi = jrd.normal(key2)
    t_sigma_rw = jrd.normal(key3)
    t_params = { "log_r": t_log_r, "alpha": t_alpha, "inv_sqrt_phi": t_inv_sqrt_phi, "sigma_rw": t_sigma_rw }
    return t_params


seed = 441_582
key = jrd.key(seed)
init_key, nuts_key = jrd.split(key, 2)
t_params_init = random_init_transformed(init_key)
print(f"{t_params_init=}")

params_init, log_adjust = inv_transform(t_params_init)
t_params_init_round_trip = transform(params_init)

print(f"{log_adjust=}")
print(f"{t_params_init_round_trip=}")


def random_markov_chain(key, kernel, init_state, num_draws):
    @jax.jit
    def one_step(state, key):
        state, _ = kernel(key, state)
        return state, state
    keys = jrd.split(key, num_draws)
    _, states = jax.lax.scan(one_step, init_state, keys)
    return states

def nuts_sample(key, log_density, init_position, num_draws):
    init_key, warmup_key, sample_key = jrd.split(key, 3)
    warmup = blackjax.window_adaptation(blackjax.nuts, log_density)
    (state, params), _ = warmup.run(warmup_key, init_position, num_steps=num_draws)
    kernel = blackjax.nuts(log_density, **params).step
    states = random_markov_chain(sample_key, kernel, state, num_draws)
    draws = states.position
    return draws

num_draws = 1_000

t_draws = nuts_sample(nuts_key, log_posterior_transformed, t_params_init, num_draws)

def inv_transform_draws(t_draws):
    draws = {
        "log_r" : t_draws["log_r"],
        "alpha": t_draws["alpha"],
        "inv_sqrt_phi": t_draws["inv_sqrt_phi"],
        "sigma_rw": jnp.exp(t_draws["sigma_rw"]),
    }
    return draws

draws = inv_transform_draws(t_draws)

import functools

posterior_means = jax.tree.map(functools.partial(jnp.mean, axis=0), draws)
posterior_stds = jax.tree.map(functools.partial(jnp.std, axis=0), draws)
print(f"{posterior_means=}")
print(f"{posterior_stds=}")
