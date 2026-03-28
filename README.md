<<<<<<< HEAD
# Covstream
Covariance matrix estimator, with formalized welford's algorithm on Lean
=======
# covstream

Minimal Lean-first scaffold for learning how to specify and prove properties of an online covariance stream estimator.

What is in the repo now:

- `Covstream/Basic.lean`: a tiny covstream model
- `Covstream/Basic.lean`: one basic theorem
- `Main.lean`: just a build sanity check

Why no `mathlib` yet:

- your `lake init ... math` attempt failed because Lake/Reservoir could not resolve `mathlib`
- you do not need `mathlib` to start learning the core modeling and proof structure

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
>>>>>>> 8df674f (adding initial commit)
