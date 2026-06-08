## a direct translation from the Stan C++ model class to JAX
## (concatenation of python code in section 6)

import jax
import jax.numpy as jnp
import jax.random as jrd
from jax.scipy import stats
import blackjax

from util import positive, real

import json
import numpy as np

def simulate_regression(n=128, p=2, n_new=4, seed=145777):
    def simulate_covariates(n):
        x = rng.normal(size=(n, p))
        x[:, 1] = x[:, 1]**2
        return x

    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.0, 5.0)
    beta = rng.normal(0.0, 2.5, size=p)
    sigma = rng.exponential(1.0 / 0.5)  # numpy rng uses scale/mean parameterization
    x = simulate_covariates(n)
    mu = alpha + x @ beta
    y = rng.normal(mu, sigma)
    x_new = simulate_covariates(n_new)
    parameters = { "alpha": alpha, "beta": beta, "sigma": sigma }

    data = {
        "N": n, "P": p, "N_new": n_new, "x": x.tolist(),
	"y": y.tolist(), "x_new": x_new.tolist(),
    "true_alpha": alpha, "true_beta": beta,
    }
    return data

data = simulate_regression()

print(f'true alpha{data["true_alpha"]}')
print(f'true beta{data["true_beta"]}')

x = jnp.array(data["x"])
y = jnp.array(data["y"])
x_new = jnp.array(data["x_new"])


## defining everything using jax.scipy densities
def log_posterior(params):
    lp = 0.0
    lp += jnp.sum(stats.norm.logpdf(params["alpha"], loc=0.0, scale=5.0))
    lp += jnp.sum(stats.norm.logpdf(params["beta"], loc=0.0, scale=2.5))
    lp += jnp.sum(stats.expon.logpdf(params["sigma"], scale=1.25))
    mu = params['alpha'] + x @ params["beta"]
    lp += jnp.sum(stats.norm.logpdf(y, loc=mu, scale=params["sigma"]))
    return lp

## need to apply transforms
def transform(params):
    t_alpha = params["alpha"]
    t_beta = jnp.array(params["beta"])
    t_sigma = jnp.log(params["sigma"])
    t_params = { "alpha": t_alpha, "beta": t_beta, "sigma": t_sigma }
    return t_params

def inv_transform(t_params):
    log_adjust = 0.0
    alpha = t_params["alpha"]
    beta = jnp.array(t_params["beta"])
    sigma = jnp.exp(t_params["sigma"])
    log_adjust += t_params["sigma"]
    params = { "alpha": alpha, "beta": beta, "sigma": sigma }
    return params, log_adjust

def log_posterior_transformed(t_params):
    params, log_adjust = inv_transform(t_params)
    log_post = log_posterior(params)
    return log_adjust + log_post

def random_init_transformed(key):
    key0, key1, key2 = jrd.split(key, 3)
    t_alpha = jrd.normal(key0)
    t_beta = jrd.normal(key1, shape=(2,))
    t_sigma = jrd.normal(key2)
    t_params = { "alpha": t_alpha, "beta": t_beta, "sigma": t_sigma }
    return t_params

seed = 441_582
key = jrd.key(seed)
init_key, nuts_key = jrd.split(key, 2)
t_params_init = random_init_transformed(init_key)
print(f"{t_params_init=}")

params_init, log_adjust = inv_transform(t_params_init)
t_params_init_round_trip = transform(params_init)

print(f"{params_init=}")
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
        "alpha": t_draws["alpha"],
        "beta": t_draws["beta"],
        "sigma": jnp.exp(t_draws["sigma"]),
    }
    return draws

draws = inv_transform_draws(t_draws)

import functools

posterior_means = jax.tree.map(functools.partial(jnp.mean, axis=0), draws)
posterior_stds = jax.tree.map(functools.partial(jnp.std, axis=0), draws)
print(f"{posterior_means=}")
print(f"{posterior_stds=}")


data_new = { "x_new": x_new }

def posterior_predictive(key, params):
    mu = params["alpha"] + data_new["x_new"] @ params["beta"]
    z = jax.random.normal(key, shape=mu.shape)
    y_new = mu + params["sigma"] * z
    return {"y_new": y_new}

def posterior_predictive_draws(key, draws):
    N = draws["alpha"].shape[0]
    keys = jax.random.split(key, N)
    return jax.vmap(posterior_predictive, in_axes=(0, 0))(keys, draws)

key, gq_key = jrd.split(key, 2)
pred_draws = posterior_predictive_draws(gq_key, draws)

posterior_pred_means = jax.tree.map(functools.partial(jnp.mean, axis=0), pred_draws)
posterior_pred_stds = jax.tree.map(functools.partial(jnp.std, axis=0), pred_draws)
print(f"{posterior_pred_means=}")
print(f"{posterior_pred_stds=}")
