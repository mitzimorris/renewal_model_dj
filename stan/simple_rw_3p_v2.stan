// simple renewal model
// infer rt,  ascertainment rate, and overdispersion
// random walk prior on time-varying renewal process
// one source of observations
// fixed estimates for infection, observational delay distributions, initial rt,
// and i0_scale which anchors infection/ascertainment rates
functions {
  real lotka_euler_residual(real r, real rt0, vector w) {
    int S = num_elements(w);
    real acc = 0;
    for (s in 1:S) {
      acc += w[s] * exp(-r * s);
    }
    return rt0 * acc - 1;
  }

  real solve_growth_rate(real rt0, vector w) {
    real lo = -2;
    real hi = 2;
    real mid;
    if (abs(rt0 - 1.0) < 1e-12) {
      return 0;
    }
    // Bisection. Residual is monotone decreasing in r.
    for (iter in 1:100) {
      mid = 0.5 * (lo + hi);
      if (lotka_euler_residual(mid, rt0, w) > 0) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return 0.5 * (lo + hi);
  }
}
data {
  // observational data
  int<lower=0> T;                // total days
  int<lower=0> S;                // generation interval length
  int<lower=0> D;                // delay distribution length

  array[T] int<lower=0> y;       // observed events

  simplex[S] w;                  // generation interval
  simplex[D] pi;                 // delay distribution

  // used to seed pre-observation trajectory
  real<lower=0> rt0;             // reproduction number at time 0
  real<lower=0> i0_scale;        // extrapolated infection level at time 0
  real<lower=0> population_size;
}
transformed data {
  int<lower=0> L = max(D, S);      // max lookback
  simplex[S] w_rev = reverse(w);   // rearrange for convolution
  simplex[D] pi_rev = reverse(pi);

  vector[L] J;
  real r0_approx = solve_growth_rate(rt0, w);
  print(r0_approx);
  for (j in 1:L) {
    int seed_time = j - L - 1;  // maps index j to pre-observation time -L, ..., -1
    J[j] = population_size * i0_scale * exp(r0_approx * seed_time);
  }
  print("J ", J);
}
parameters {
  vector[T] log_r;                // time-varying reproduction number (log scale)
  real<lower=0, upper=1> alpha;   // ascertainment rate
  real<lower=0> inv_sqrt_phi;     // 0 = Poisson; 1 = heavy overdispersion
  real<lower=0> sigma_rw;         // temporal smoothness
}
transformed parameters {
  real<lower=0> phi = inv(square(inv_sqrt_phi));
}
model {
  vector[T] r = exp(log_r);
  vector[T] mu;
  vector[L + T] I;
  I[1:L] = J;
  for (t in 1:T) {
    int lpt = L + t;
    I[lpt] = r[t] * dot_product(I[lpt - S:lpt - 1], w_rev);
    mu[t] = alpha * dot_product(I[lpt - D + 1:lpt], pi_rev);
  }
  y ~ neg_binomial_2(mu, phi);

  // priors on all parameters
  alpha ~ beta(1, 100);
  inv_sqrt_phi ~ normal(0, 1);
  log_r[1] ~ normal(log(rt0), 0.1);
  log_r[2:T] ~ normal(log_r[1:T-1], sigma_rw);  // RW1 prior on r
  sigma_rw ~ normal(0, 0.1);
}
generated quantities {
  vector[T] r = exp(log_r);
}  
