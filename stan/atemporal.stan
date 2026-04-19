// simple renewal model
// one source of observations
// fixed estimates for infection, observational delay distributions
data {
  // observational data
  int<lower=0> T;                // total days
  int<lower=0> S;                // generation interval length
  int<lower=0> D;                // delay distribution length
  array[T] int<lower=0> y;       // observed events

  // estimated params - move to params for Rubin-style
  simplex[S] w;                  // generation interval
  simplex[D] pi;                 // delay distribution
  vector<lower=0>[max(D, S)] J;  // infections at time t < 0
}
transformed data {
  int<lower=0> L = max(D, S);      // max lookback
  simplex[S] w_rev = reverse(w);   // required for convolution
  simplex[D] pi_rev = reverse(pi);
  real eps = 1e-9;
}
parameters {
  vector<lower=0>[T] r;
  real<lower=0, upper=1> alpha;
  real<lower=0> phi;
}
model {
  vector[L + T] I;
  I[1:L] = J;
  vector[T] mu;
  for (t in 1:T) {
    int lpt = L + t;
    I[lpt] = r[t] * dot_product(I[lpt - S:lpt - 1], w_rev);
    mu[t] = alpha * dot_product(I[lpt - D:lpt - 1], pi_rev) + eps;
  }
  alpha ~ beta(1, 10);            // skewed towards small values
  r ~ lognormal(0, 0.5);          // R centered at 1 - no temporal trends
  phi ~ lognormal(log(50), 0.4);  // median concentration - expect less overdispersion
  y ~ neg_binomial_2(mu, phi);
}
