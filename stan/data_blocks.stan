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
