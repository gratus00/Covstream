# Rust API Mapping

This document describes how the Lean-side contract maps onto the current Rust
crate surface.

## Public Rust Entry Points

- `CovstreamState`
  User-facing wrapper that stores a default output layout and exposes the
  simplest ingest and extraction API
- `CovstreamCore`
  Lower-level engine with explicit extraction methods for each layout
- `MatrixLayout`
  Output layout selector: `RowMajor` or `UpperTrianglePacked`
- `ShrinkageMode`
  Shrinkage coefficient policy: `FixedAlpha(f64)` or `ClippedAlpha(f64)`
- `CovstreamError`
  Runtime validation and extraction errors

## Lean-To-Rust Mapping

- `Covstream.StreamingContract`
  Rust: `CovstreamState`
- `Covstream.CovstreamError`
  Rust: `enum CovstreamError`
- `Covstream.MatrixLayout`
  Rust: `enum MatrixLayout`
- `Covstream.ShrinkageMode`
  Rust: `enum ShrinkageMode`

## Function Mapping

- `StreamingContract.init?`
  Rust: `CovstreamState::new(dimension, layout) -> Result<CovstreamState, CovstreamError>`
- `StreamingContract.observe?`
  Rust: `state.observe(sample: &[f64]) -> Result<(), CovstreamError>`
- `StreamingContract.observeBatch?`
  Rust: `state.observe_batch_row_major(samples: &[f64]) -> Result<(), CovstreamError>`
- `StreamingContract.observeTrusted?`
  Rust: `state.observe_trusted_finite(sample: &[f64]) -> Result<(), CovstreamError>`
- `StreamingContract.observeBatchTrusted?`
  Rust: `state.observe_batch_row_major_trusted_finite(samples: &[f64]) -> Result<(), CovstreamError>`
- `StreamingContract.covarianceRowMajor?`
  Rust: `state.covariance_row_major() -> Result<Vec<f64>, CovstreamError>`
- `StreamingContract.covarianceBuffer?`
  Rust: `state.covariance_buffer() -> Result<Vec<f64>, CovstreamError>`
- `StreamingContract.ledoitWolfRowMajor?`
  Rust: `state.ledoit_wolf_row_major(mode) -> Result<Vec<f64>, CovstreamError>`
- `StreamingContract.ledoitWolfBuffer?`
  Rust: `state.ledoit_wolf_buffer(mode) -> Result<Vec<f64>, CovstreamError>`

## Additional Rust-Only Operational APIs

These APIs are implementation-facing conveniences rather than part of the
minimal mathematical contract:

- `CovstreamCore::merge`
  combine two same-dimension partial states
- `observe_batch_row_major_parallel`
  parallel reduction path for sufficiently large row-major batches
- `*_into(&mut [f64])`
  write into caller-provided buffers to avoid repeated allocation
- `reset()`
  reuse an existing state without reallocating internal buffers

## Buffer Contract

- `MatrixLayout::RowMajor`
  output length is `k * k`
- `MatrixLayout::UpperTrianglePacked`
  output length is `k * (k + 1) / 2`

These correspond to the Lean-side encoding lemmas for matrix layout length and
shape preservation.

## Runtime Rules

- reject `dimension = 0`
- reject samples whose length differs from `dimension`
- reject malformed flat row-major batch buffers
- reject `NaN` and infinite inputs on checked ingest paths
- reject covariance and shrinkage extraction before `sample_count >= 2`
- clamp shrinkage coefficients in `ClippedAlpha`

## Proof Boundary

The Lean side proves the real-number specification and output-shape contract.
The Rust side implements the same algorithmic structure over `f64`, adds
validation around external inputs, and ships tests and benchmarks for the
concrete implementation.

The current crate does not claim a complete floating-point refinement proof from
Rust `f64` execution to Lean `Real`.
