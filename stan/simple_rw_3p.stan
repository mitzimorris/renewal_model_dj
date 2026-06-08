#include solve_growth_rate.stan
data {
  // observational data
  int<lower=3> T;                // total days
  int<lower=0> S;                // generation interval length
  int<lower=0> D;                // delay distribution length

  array[T] int<lower=0> y;       // observed events

  simplex[S] w;                  // generation interval
  simplex[D] pi;                 // delay distribution

  // used to seed pre-observation trajectory
  real<lower=1e-12> rt0;         // reproduction number at time 0
  real<lower=0> i0_scale;        // extrapolated infection level at time 0
  real<lower=0> population_size;
}
transformed data {
  simplex[S] w_rev = reverse(w); // rearrange for convolution
  simplex[D] pi_rev = reverse(pi);
  int<lower=0> L = max(D, S); // max lookback
  vector[L] J;
  real r0_approx = solve_growth_rate(rt0, w);
  for (j in 1:L) {
    int seed_time = j - L - 1;  // maps index j to pre-observation time -L, ..., -1
    J[j] = population_size * i0_scale * exp(r0_approx * seed_time);
  }
}
parameters {
  vector[T] log_r;                // time-varying reproduction number (log scale)
  real<lower=0, upper=1> alpha;   // ascertainment rate
  real<lower=0> inv_sqrt_phi;     // 0 = Poisson; 1 = heavy overdispersion
  real<lower=0> sigma_rw;         // temporal smoothness
}
transformed parameters {
  real<lower=0> phi = inv(square(inv_sqrt_phi));
  vector[T] r = exp(log_r);
}
model {
  vector[T] mu;
  vector[L + T] I;
  I[1:L] = J;
  for (t in 1:T) {   // auto-regressive renewal process
    int lpt = L + t;
    I[lpt] = r[t] * dot_product(I[lpt - S:lpt - 1], w_rev);
  }
  for (t in 1:T) {   // observation process - convole infection with delay distribution
    int lpt = L + t;
    mu[t] = alpha * dot_product(I[lpt - D + 1:lpt], pi_rev);
  }
  y ~ neg_binomial_2(mu, phi);

  // priors on all parameters
  alpha ~ beta(1, 100);
  inv_sqrt_phi ~ normal(0, 1);
  log_r[1] ~ normal(0, 0.5);          // initial R centered at 1
  log_r[2:T] ~ normal(log_r[1:T-1], sigma_rw);  // RW1 prior on r
  sigma_rw ~ normal(0, 0.5);
}
