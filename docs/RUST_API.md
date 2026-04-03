# Rust API Mapping

This document is the intended translation from the Lean contract layer to the
Rust implementation layer.

## Core Mapping

- `Covstream.StreamingContract`
  Rust: `CovstreamState`
- `Covstream.CovstreamError`
  Rust: `enum CovstreamError`
- `Covstream.MatrixLayout`
  Rust: `enum MatrixLayout`
- `Covstream.ShrinkageMode`
  Rust: `enum ShrinkageMode`

## Rust Shape

```rust
pub struct CovstreamState {
    dimension: usize,
    layout: MatrixLayout,
    sample_count: usize,
    mean: Vec<f64>,
    cov_numerator: Vec<f64>,
}

pub enum CovstreamError {
    ZeroDimension,
    WrongDimension { expected: usize, got: usize },
    InsufficientSamples { actual: usize },
    NonFiniteInput,
}

pub enum MatrixLayout {
    RowMajor,
    UpperTrianglePacked,
}

pub enum ShrinkageMode {
    FixedAlpha(f64),
    ClippedAlpha(f64),
}
```

## Function Mapping

- `StreamingContract.init?`
  Rust: `CovstreamState::new(dimension, layout) -> Result<CovstreamState, CovstreamError>`
- `StreamingContract.observe?`
  Rust: `state.observe(sample: &[f64]) -> Result<(), CovstreamError>`
- `StreamingContract.covariance?`
  Rust: `state.covariance_matrix() -> Result<Vec<f64>, CovstreamError>`
- `StreamingContract.ledoitWolf?`
  Rust: `state.ledoit_wolf_matrix(mode) -> Result<Vec<f64>, CovstreamError>`
- `StreamingContract.covarianceBuffer?`
  Rust: `state.covariance_buffer() -> Result<Vec<f64>, CovstreamError>`
- `StreamingContract.ledoitWolfBuffer?`
  Rust: `state.ledoit_wolf_buffer(mode) -> Result<Vec<f64>, CovstreamError>`

## Buffer Contract

- `MatrixLayout::RowMajor`
  Output length must be `k * k`
- `MatrixLayout::UpperTrianglePacked`
  Output length must be `k * (k + 1) / 2`

These are formalized in:

- `StreamingContract.encodeRowMajor_length`
- `StreamingContract.encodeUpperTrianglePacked_length`
- `StreamingContract.encodeMatrix_length`

## Runtime Rules

- Reject `dimension = 0`
- Reject samples whose length differs from `dimension`
- Reject NaN and infinite inputs
- Reject covariance extraction before `sample_count >= 2`
- Clamp shrinkage coefficients in `ClippedAlpha`

## Proof Alignment

The Lean side proves:

- checked extraction returns the exact mathematical object encoded in the chosen layout
- buffer lengths match the declared layout contract
- clipped shrinkage is stable under entrywise perturbations

The Rust side should mirror the contract exactly, then add `f64`-specific tests
and benchmarks on top.
