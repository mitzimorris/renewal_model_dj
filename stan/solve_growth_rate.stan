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
