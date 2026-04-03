# Covstream

`Covstream` is a Lean 4 formalization of:

1. an exact `Real`-valued streaming covariance estimator based on Welford's algorithm
2. Ledoit-Wolf style linear shrinkage toward the scaled identity target `μ I`
3. the Frobenius-loss optimization story behind the shrinkage coefficient
4. a checked runtime contract that a Rust implementation can mirror directly
5. first perturbation bounds for future floating-point refinement

The intended project architecture is:

- Lean: exact mathematical specification and theorems
- Rust: fast `f64` implementation built against this specification

## If You Only Read Three Things

1. [Covstream/Welford.lean](./Covstream/Welford.lean)
   This contains the core streaming-to-batch correctness theorem.
2. [Covstream/Contract.lean](./Covstream/Contract.lean)
   This is the API contract the Rust implementation should mirror.
3. [docs/RUST_API.md](./docs/RUST_API.md)
   This is the direct Lean-to-Rust naming and behavior map.

## Two-Minute Reading Guide

If you are reviewing the project quickly, read these in order:

1. [Covstream.lean](./Covstream.lean)
   The top-level map of the formalization.
2. [Covstream/Welford.lean](./Covstream/Welford.lean)
   Streaming covariance state and the proof that the streaming estimator matches
   the classical batch covariance formula.
3. [Covstream/LedoitWolf.lean](./Covstream/LedoitWolf.lean)
   Shrinkage target, shrinkage operator, and symmetry / PSD preservation.
4. [Covstream/ShrinkageOptimization.lean](./Covstream/ShrinkageOptimization.lean)
   The Frobenius-loss argument showing which shrinkage coefficient is optimal.
5. [Covstream/Contract.lean](./Covstream/Contract.lean)
   The checked API a Rust or C++ implementation should mirror.
6. [Covstream/ErrorBounds.lean](./Covstream/ErrorBounds.lean)
   First perturbation bounds for future floating-point refinement work.
7. [Covstream/Examples.lean](./Covstream/Examples.lean)
   Concrete 2D examples of the checked API.
8. [docs/RUST_API.md](./docs/RUST_API.md)
   The intended Rust translation of the contract layer.

## Main Theorems

The four most important statements are:

1. `fitCovariance_eq_classicalCovariance`
   In `Covstream.Welford`, the streaming Welford covariance equals the direct
   classical sample covariance from the same observation history.
2. `fitLedoitWolf_eq_classicalLedoitWolf`
   In `Covstream.LedoitWolf`, the streaming covariance pipeline and the direct
   classical covariance pipeline produce the same shrunk estimator.
3. `clippedCoeff_minimizes_shrinkageLoss_on_unitInterval`
   In `Covstream.ShrinkageOptimization`, the clipped coefficient is optimal
   among all valid coefficients `α ∈ [0,1]`.
4. `oracleLedoitWolfCoeff_minimizes_shrinkageLoss_on_unitInterval`
   In `Covstream.ShrinkageOptimization`, the oracle Ledoit-Wolf coefficient is
   optimal for the scaled-identity target under Frobenius loss.

## Module Layout

- [Covstream/Welford.lean](./Covstream/Welford.lean)
  Exact streaming state, history replay, covariance vocabulary, and the
  streaming-to-batch correspondence theorem.
- [Covstream/LedoitWolf.lean](./Covstream/LedoitWolf.lean)
  Structural shrinkage theory and history-facing API definitions.
- [Covstream/FrobeniusBasic.lean](./Covstream/FrobeniusBasic.lean)
  Basic matrix arithmetic and Frobenius norm identities.
- [Covstream/ShrinkageOptimization.lean](./Covstream/ShrinkageOptimization.lean)
  Quadratic shrinkage loss, unconstrained optimum, interval-clipped optimum,
  and oracle Ledoit-Wolf coefficient theorems.
- [Covstream/FrobeniusLoss.lean](./Covstream/FrobeniusLoss.lean)
  Backward-compatible umbrella import for the optimization layer.
- [Covstream/Contract.lean](./Covstream/Contract.lean)
  Checked constructor/update/extraction API with explicit errors, layout choices,
  and boundary theorems for the implementation layer.
- [Covstream/ErrorBounds.lean](./Covstream/ErrorBounds.lean)
  Abstract entrywise perturbation bounds for target formation and shrinkage.
- [Covstream/Examples.lean](./Covstream/Examples.lean)
  Small concrete examples of successful initialization, rejected updates, and
  exact buffer extraction.
- [Covstream/Basic.lean](./Covstream/Basic.lean)
  Backward-compatible umbrella import for the whole formalization.
- [docs/RUST_API.md](./docs/RUST_API.md)
  A direct map from Lean boundary types and functions to Rust names.

## What This Formalization Claims

In the exact `Real` model, the code proves:

- Welford's streaming covariance state matches the classical batch covariance
- linear shrinkage toward `μ I` preserves symmetry
- linear shrinkage toward `μ I` preserves PSD under the expected assumptions
- the oracle shrinkage coefficient is optimal for Frobenius loss on the valid
  interval `[0,1]`
- the runtime-facing contract makes dimension checks, sample-count checks, and
  matrix layout choices explicit
- perturbation lemmas bound how covariance and shrinkage outputs move under
  entrywise input error
- the checked API admits small concrete examples with exact buffer outputs

## Useful Commands

```bash
source ~/.elan/env
lake build
lake env lean Covstream/Welford.lean
lake env lean Covstream/LedoitWolf.lean
lake env lean Covstream/ShrinkageOptimization.lean
lake env lean Covstream/Contract.lean
lake env lean Covstream/ErrorBounds.lean
lake env lean Covstream/Examples.lean
lake exe covstream
```
