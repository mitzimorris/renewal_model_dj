// simple renewal model
// one source of observations
// fixed estimates for infection, observational delay distributions
data {
  int<lower=0> T;
  int<lower=0> S;
  int<lower=0> D;
  array[T] int<lower=0> y;
  // estimated params - move to params for Rubin-style
  simplex[S] w;
  simplex[D] pi;
  vector<lower=0>[max(D, S)] J;
}
transformed data {
  simplex[S] w_rev = reverse(w);
  simplex[D] pi_rev = reverse(pi);
  int<lower=0> L = max(D, S);
}
parameters {
  vector<lower=0>[T] r;
  real<lower=0, upper=1> alpha;
  real<lower=0> nu;
}
model {
  vector[L + T] I;
  I[1:L] = J;
  vector[T] mu;
  for (t in 1:T) {
    int lpt = L + t;
    I[lpt] = r[t] * dot_product(I[lpt - S:lpt - 1], w_rev);
    mu[t] = alpha * dot_product(I[lpt - D:lpt - 1], pi_rev);
  }
  y ~ neg_binomial(mu, nu);
}
