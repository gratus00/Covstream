# Covstream
Covariance matrix estimator, with formalized welford's algorithm on Lean
=======
# covstream

Covariance matrix estimator, with formalized welford's algorithm on Lean

Useful commands:

```bash
lake build
lake env lean Covstream/Basic.lean
lake exe covstream
```

Suggested learning path:

1. Keep the Lean model simple and prove a few properties first.
2. Once the spec feels stable, add a small Rust crate that mirrors the Lean definitions.
3. Then decide how strict you want the connection to be between the Lean proof and the Rust code.
