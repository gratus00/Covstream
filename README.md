# Covstream

`Covstream` is a Lean-backed Rust library for fixed-dimension streaming
covariance and Ledoit-Wolf shrinkage over `f64`.

It is aimed at portfolio analytics, telemetry analytics, numerical experiments,
and other systems that need to keep a covariance estimate hot in memory as new
aligned vectors arrive.

Use it when you want to:

- update covariance one aligned sample at a time without rebuilding from scratch
- ingest flat row-major batches for backtests or research jobs
- extract either dense row-major or packed symmetric outputs
- keep the mathematical story tied to a Lean specification without turning the
  Rust crate into a research prototype

## Installation

After publishing, add the crate with:

```bash
cargo add covstream
```

Or put it directly in `Cargo.toml`:

```toml
[dependencies]
covstream = "0.1"
```

## At A Glance

- Use `CovstreamState` unless you need low-level control. It is the default
  user-facing API.
- Use `UpperTrianglePacked` when you care about throughput or memory. Symmetric
  covariance matrices do not need dense storage.
- Safe ingest APIs reject malformed shapes and `NaN` / `Inf`. The
  `trusted_finite` APIs are faster, but they assume upstream validation.
- The public crate is intentionally small: numerical state, batch ingest,
  extraction, shrinkage, examples, tests, benchmarks, and CI.

## What Ships In `0.1.2`

Today the Rust crate supports:

- fixed-dimension streaming updates via `observe(&[f64])`
- flat batch ingest via `observe_batch_row_major(&[f64])`
- optional parallel batch ingest for large row-major batches
- optional trusted-finite ingest paths for validated pipelines
- Welford-style covariance accumulation
- covariance extraction in:
  - full row-major layout
  - packed upper-triangle layout
- Ledoit-Wolf style linear shrinkage toward the scaled-identity target `μI`
- reusable `*_into(&mut [f64])` extraction methods
- state reuse via `reset()`
- a small but useful release boundary instead of an overgrown “kitchen sink”

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

If you are integrating `covstream` into a larger application, the usual shape
is:

1. create one fixed-dimension state per stream, portfolio, or universe
2. call `observe` for live updates or a batch ingest method for offline work
3. extract covariance or shrunk covariance only when a downstream consumer
   actually needs a snapshot

## Typical Use Cases

Common ways a developer would use `covstream`:

- live market or telemetry streams:
  call `observe(&sample)` each time a new aligned vector arrives
- offline backtests or research jobs:
  load a flat row-major batch and call `observe_batch_row_major(&batch)`
- larger batch analytics jobs:
  use `observe_batch_row_major_parallel(&batch)` when the batch is large enough
  that parallel reduction is worth the overhead
- downstream optimizers and risk engines:
  extract packed covariance buffers to avoid dense symmetric storage when the
  consumer can work with packed layout

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

For large offline or micro-batched workloads, `observe_batch_row_major_parallel`
and `observe_batch_row_major_parallel_trusted_finite` can reduce total ingest
time by merging per-task partial states. For small batches, the serial path is
usually better because task scheduling overhead dominates.

## Security And Safety Notes

- The library API does not perform network I/O, file I/O, or subprocess
  execution.
- Checked ingest paths reject malformed shapes and non-finite values.
- `trusted_finite` methods are intended only for callers that already validate
  input upstream.
- The only `unsafe` code is isolated to optional AArch64 SIMD leaf kernels.
- A minimal security policy is in [SECURITY.md](./SECURITY.md).

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

## Non-Claims

`covstream` is intentionally narrower than a full analytics platform. The `0.1`
crate does **not** claim:

- a formal floating-point refinement proof from Rust `f64` execution to Lean
  `Real`
- dynamically changing asset dimensions inside a single state object
- CSV, socket, market-data, or ticker ingestion inside the core crate
- production order-routing, exchange connectivity, or execution-system features

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

Each figure below is the median time for one full benchmarked call named in the
label, not for one arithmetic primitive inside that call.

Representative 256-dimension medians from two local hosts:

- **MacBook Air (Apple M2, 8 CPU cores, 16 GB RAM)** on **April 11, 2026**
- **Linux desktop (AMD Ryzen 5 5600, 6 cores / 12 threads, 32 GB RAM)** on
  **April 12, 2026**

| Benchmark | MacBook Air M2 | Linux desktop Ryzen 5 5600 |
| --- | ---: | ---: |
| `observe_hot/trusted/256` | `11.4 µs` | `8.25 µs` |
| `covariance_extract/packed_into/256` | `12.7 µs` | `5.63 µs` |
| `covariance_extract/row_major_into/256` | `85.1 µs` | `78.2 µs` |
| `shrinkage_extract/ledoit_wolf_packed_into/256` | `15.5 µs` | `6.98 µs` |
| `observe_batch/batch_call/d256_n256` | `3.92 ms` | `2.15 ms` |
| `observe_batch_parallel/trusted_serial/d256_n1024` | `21.6 ms` | `8.41 ms` |
| `observe_batch_parallel/trusted_parallel/d256_n1024` | `8.31 ms` | `1.82 ms` |

For very small fixed dimensions, the Linux desktop also shows single-update
medians in the tens to hundreds of nanoseconds:

- `observe/2`: about `48.2 ns`
- `observe/8`: about `98.3 ns`
- `observe/32`: about `348 ns`
- `observe/64`: about `849 ns`

These `observe/k` figures are useful as a small-d latency hook, but the
hot-state `observe_hot/trusted/256` benchmark is more representative once the
state already exists and the workload is not dominated by setup.

The Linux desktop run also makes the batch-parallel crossover clearer:

- `d32_n256`: essentially break-even, with parallel overhead offsetting the
  extra worker utilization
- `d32_n1024`: `trusted_serial` about `206 µs`, `trusted_parallel` about `60.4 µs`
- `d128_n256`: `trusted_serial` about `545 µs`, `trusted_parallel` about `202 µs`
- `d256_n1024`: `trusted_serial` about `8.41 ms`, `trusted_parallel` about `1.82 ms`
- `d512_n1024`: `trusted_serial` about `34.0 ms`, `trusted_parallel` about `8.44 ms`

The important usage patterns are:

- packed output is cheaper than full row-major output
- `*_into` methods avoid repeated output allocations
- batch ingest improves API ergonomics and can reduce validation overhead
- parallel batch ingest is mainly worthwhile for large `n` and moderate-to-large `k`
- the main runtime cost is still the `O(k^2)` covariance update itself

Run benchmarks with:

```bash
cargo bench
cargo bench --bench core_bench -- observe_batch_parallel
cargo bench --bench core_bench -- d256_n1024
```

## Roadmap

Near-term follow-ups that fit the current project direction:

- docs.rs-facing examples and more benchmark notes for common `k`/`n` regimes
- a thin companion transport layer or daemon example for live socket ingestion,
  kept outside the core crate
- additional shrinkage research and estimator comparisons when there is clear
  user demand
- future floating-point refinement work against the Lean specification
  boundary

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
lake exe cache get
lake build
lake env lean Covstream/Welford.lean
lake env lean Covstream/LedoitWolf.lean
lake env lean Covstream/ShrinkageOptimization.lean
lake env lean Covstream/Contract.lean
```

If a fresh machine reports missing `ProofWidgets` assets, `lake exe cache get`
usually fixes it. If you hit an `incompatible header` error from stale `.olean`
artifacts, run `lake clean` and build again.
