# Covstream

`Covstream` is a fixed-dimension streaming covariance library with Lean-backed
specifications and a fast Rust `f64` implementation.

It is aimed at quant research, portfolio analytics, and other systems that need
to keep a covariance estimate updated as new aligned vectors arrive.

## If You Only Read Three Things

- Use `CovstreamState` unless you need low-level control. It is the default
  user-facing API.
- Use `UpperTrianglePacked` when you care about throughput or memory. Symmetric
  covariance matrices do not need dense storage.
- Safe ingest APIs reject malformed shapes and `NaN` / `Inf`. The
  `trusted_finite` APIs are faster, but they assume upstream validation.

## What The Rust Library Does

Today the Rust crate supports:

- fixed-dimension streaming updates via `observe(&[f64])`
- flat batch ingest via `observe_batch_row_major(&[f64])`
- optional trusted-finite ingest paths for validated pipelines
- Welford-style covariance accumulation
- covariance extraction in:
  - full row-major layout
  - packed upper-triangle layout
- Ledoit-Wolf style linear shrinkage toward the scaled-identity target `μI`
- reusable `*_into(&mut [f64])` extraction methods
- state reuse via `reset()`

The main public types are:

- `CovstreamState`
  Recommended wrapper that remembers a preferred output layout
- `CovstreamCore`
  Lower-level engine with explicit control over extraction paths
- `MatrixLayout`
  `RowMajor` or `UpperTrianglePacked`
- `ShrinkageMode`
  `FixedAlpha(f64)` or `ClippedAlpha(f64)`
- `CovstreamError`
  Explicit runtime errors for invalid dimensions, malformed batches,
  non-finite input, insufficient samples, and undersized output buffers

## Choosing The API

Use `CovstreamState` when you want the simplest interface:

```rust
use covstream::{CovstreamState, MatrixLayout, ShrinkageMode};

let mut state = CovstreamState::new(3, MatrixLayout::UpperTrianglePacked)?;
state.observe(&[0.0010, -0.0005, 0.0007])?;
state.observe(&[0.0008, -0.0003, 0.0004])?;
state.observe(&[-0.0006, 0.0002, -0.0001])?;

let covariance = state.covariance_buffer()?;
let shrunk = state.ledoit_wolf_buffer(ShrinkageMode::ClippedAlpha(0.20))?;
# Ok::<(), covstream::CovstreamError>(())
```

Use `CovstreamCore` when you want direct control over row-major vs packed
extraction and lower-level ingest methods.

## Batch Ingest

`observe_batch_row_major` expects a flat slice of back-to-back samples.

For dimension `3`, this means:

```text
[
  x00, x01, x02,
  x10, x11, x12,
  x20, x21, x22,
]
```

Example:

```rust
use covstream::{CovstreamState, MatrixLayout};

let mut state = CovstreamState::new(3, MatrixLayout::UpperTrianglePacked)?;

let batch = [
    0.0010, -0.0005, 0.0007,
    0.0008, -0.0003, 0.0004,
   -0.0006,  0.0002, -0.0001,
];

state.observe_batch_row_major(&batch)?;
# Ok::<(), covstream::CovstreamError>(())
```

## Examples

The repository ships with three example programs:

- [examples/streaming_covariance.rs](./examples/streaming_covariance.rs)
  Small readable walkthrough of ingest, covariance, shrinkage, packed output,
  and reset
- [examples/batch_returns.rs](./examples/batch_returns.rs)
  CSV-like aligned returns input parsed into a flat batch buffer
- [examples/throughput.rs](./examples/throughput.rs)
  Synthetic high-volume stream for rough throughput measurement

Run them with:

```bash
cargo run --example streaming_covariance
cargo run --example batch_returns
cargo run --release --example throughput -- 64 100000
```

## Initial Public Release Boundary

The current public crate boundary is **`0.1.0`**.

That release includes:

- fixed-dimension streaming covariance over `f64`
- safe and trusted-finite ingest APIs
- flat batch ingest
- row-major and packed upper-triangle extraction
- Ledoit-Wolf style linear shrinkage toward `μI`
- reusable output buffers via `*_into`
- examples, integration tests, Criterion benchmarks, and CI

It does not yet claim:

- a formal floating-point refinement proof from Rust `f64` to Lean `Real`
- dynamically changing asset dimensions inside one state object
- CSV/network/ticker ingestion inside the core crate itself

## Numerical Contract

The repository has two layers:

- Lean 4 + Mathlib formalization of the mathematical model
- Rust implementation over `f64`

The practical contract is:

- Lean proves the real-number specification of the streaming covariance and
  shrinkage pipeline.
- Rust implements the same algorithmic structure using `f64`.
- Safe Rust ingest paths reject malformed inputs and non-finite values.
- Covariance and shrinkage extraction require at least two samples.

For finance and risk analytics, `f64` is the normal numerical regime here.
Covstream is estimating statistical structure, not storing exact money amounts.

The current Rust crate does **not** claim a formal floating-point refinement
proof against the Lean `Real` model yet. The Lean side provides the exact target
specification and checked output contract that a future floating-point analysis
can refine against.

## Why Shrinkage Matters

Raw sample covariance can be noisy, especially when the number of assets is
large relative to the number of observations.

Covstream exposes Ledoit-Wolf style linear shrinkage toward the scaled-identity
target:

```text
Σ_shrunk = (1 - α) Σ_sample + α μI
```

where:

- `Σ_sample` is the sample covariance estimate
- `μ` is the average diagonal variance
- `I` is the identity matrix
- `α` is the shrinkage coefficient

This stabilizes the matrix while preserving the streaming update path.

## Performance Notes

`Covstream` ships with Criterion benchmarks in
[benches/core_bench.rs](./benches/core_bench.rs).

On a local MacBook Air benchmark run on **April 10, 2026**, representative
256-dimension medians were:

- `observe/256`: about `17.4 µs`
- `covariance_extract/packed_into/256`: about `11.0 µs`
- `covariance_extract/row_major_into/256`: about `99.9 µs`
- `shrinkage_extract/ledoit_wolf_packed_into/256`: about `32.5 µs`
- `observe_batch/batch_call/d256_n256`: about `4.36 ms` total for 256 samples

The important usage patterns are:

- packed output is cheaper than full row-major output
- `*_into` methods avoid repeated output allocations
- batch ingest improves API ergonomics and can reduce validation overhead
- the main runtime cost is still the `O(k^2)` covariance update itself

Run benchmarks with:

```bash
cargo bench
```

## Lean-Backed Specification

The Rust crate sits on top of a Lean formalization in the same repository.

Key Lean files:

- [Covstream/Welford.lean](./Covstream/Welford.lean)
- [Covstream/LedoitWolf.lean](./Covstream/LedoitWolf.lean)
- [Covstream/ShrinkageOptimization.lean](./Covstream/ShrinkageOptimization.lean)
- [Covstream/Contract.lean](./Covstream/Contract.lean)

Rust/Lean bridge notes:

- [docs/RUST_API.md](./docs/RUST_API.md)

## Repository Layout

- [src/lib.rs](./src/lib.rs)
  Public crate exports
- [src/core.rs](./src/core.rs)
  Low-level streaming engine
- [src/state.rs](./src/state.rs)
  User-facing layout-aware wrapper
- [src/shrinkage.rs](./src/shrinkage.rs)
  Shrinkage helpers and `ShrinkageMode`
- [examples/](./examples)
  Runnable usage examples
- [tests/](./tests)
  Integration tests
- [benches/core_bench.rs](./benches/core_bench.rs)
  Criterion benchmark suite

## Development Commands

Rust:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo bench
```

Lean:

```bash
source ~/.elan/env
lake build
lake env lean Covstream/Welford.lean
lake env lean Covstream/LedoitWolf.lean
lake env lean Covstream/ShrinkageOptimization.lean
lake env lean Covstream/Contract.lean
```
