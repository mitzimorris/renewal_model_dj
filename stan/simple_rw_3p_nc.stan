// non-centered parameterization of renewal process model
// infer rt,  ascertainment rate, and overdispersion
// random walk prior on time-varying renewal process
// one source of observations
// fixed estimates for infection, observational delay distributions,
// initial rt and i0_scale
#include solve_growth_rate.stan
#include data_blocks.stan
parameters {
  // Non-centered parameterization works on the log scale because
  // log R(t) is unconstrained; R(t) > 0 makes additive innovations awkward.
  real log_r_init; // initial level of log R(t)
  vector[T - 1] z; // standardized innovations
  real<lower=0> sigma_rw; // RW innovation SD; controls smoothness of log R(t)
  
  real<lower=0, upper=1> alpha; // ascertainment rate
  real<lower=0> inv_sqrt_phi; // 0 = Poisson; 1 = heavy overdispersion
}
transformed parameters {
  real<lower=0> phi = inv(square(inv_sqrt_phi));
}
model {
  // Reconstruct log R(t) from initial level + scaled cumulative innovations.
  vector[T] log_r;
  log_r[1] = log_r_init;
  log_r[2 : T] = log_r_init + sigma_rw * cumulative_sum(z);
  
  vector[T] mu;
  vector[L + T] I;
  I[1 : L] = J;
  for (t in 1 : T) {
    int lpt = L + t;
    I[lpt] = exp(log_r[t]) * dot_product(I[lpt - S : lpt - 1], w_rev);
    mu[t] = alpha * dot_product(I[lpt - D + 1 : lpt], pi_rev);
  }
  y ~ neg_binomial_2(mu, phi);
  
  // Priors
  alpha ~ beta(1, 100); // low ascertainment rate
  inv_sqrt_phi ~ normal(0, 1);
  log_r_init ~ normal(0, 0.5); // initial R centered at 1
  z ~ std_normal();
  sigma_rw ~ normal(0, 0.5); // half-normal
}
generated quantities {
  vector[T] r;
  {
    vector[T] log_r;
    log_r[1] = log_r_init;
    log_r[2 : T] = log_r_init + sigma_rw * cumulative_sum(z);
    r = exp(log_r);
  }
}

