import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Auxiliary functions for this model (Stan functions block)

def solve_initial_growth_rate(
    reproduction_number: float,
    generation_interval_pmf: ArrayLike,
    num_newton_steps:int = 5,
) -> ArrayLike:
    """
    Approximate the geometric growth rate r implied by a fixed
    reproduction number R and a discrete generation-interval PMF g
    """
    generation_days = jnp.arange(1, generation_interval_pmf.shape[0] + 1)

    mean_generation_day = jnp.sum(generation_interval_pmf * generation_days)
    growth_rate = (reproduction_number - 1) / (
        reproduction_number * mean_generation_day
    )

    def newton_step(growth_rate, _):
        exp_terms = jnp.exp(-growth_rate * generation_days)

        neg_mgf = jnp.sum(generation_interval_pmf * exp_terms)
        neg_mgf_derivative = jnp.sum(
            generation_interval_pmf * (-generation_days) * exp_terms
        )

        residual = reproduction_number * neg_mgf - 1
        residual_derivative = reproduction_number * neg_mgf_derivative

        growth_rate_next = growth_rate - residual / residual_derivative
        return growth_rate_next, None

    growth_rate, _ = jax.lax.scan(
        newton_step,
        growth_rate,
        xs=None,
        length=num_newton_steps,
    )

    return growth_rate
