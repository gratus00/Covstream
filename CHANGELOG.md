# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- added a repo-root MIT `LICENSE` file
- README benchmark notes now include representative Linux desktop medians and
  small-dimension nanosecond figures
- README now documents `lake exe cache get` and `lake clean` for fresh Lean setups
- crate packaging excludes the local `.codex` workspace file

## [0.1.0] - 2026-04-10

Initial public Rust crate boundary.

### Added

- `CovstreamCore` low-level streaming covariance engine
- `CovstreamState` layout-aware wrapper API
- safe single-sample ingest via `observe`
- safe flat batch ingest via `observe_batch_row_major`
- optional parallel flat batch ingest via partial-state reduction
- trusted-finite ingest APIs for validated pipelines
- `merge` support for sharded or parallel accumulation workflows
- covariance extraction in row-major and packed upper-triangle layouts
- Ledoit-Wolf style linear shrinkage toward `μI`
- reusable `*_into` extraction methods to avoid repeated allocations
- `reset()` for state reuse without reallocating buffers
- AArch64 SIMD leaf kernels for selected vector operations
- runnable examples:
  - `streaming_covariance`
  - `batch_returns`
  - `throughput`
- integration tests and Criterion benchmarks
- Rust CI for format, clippy, and test gates
- Lean-backed repository documentation and Rust/Lean bridge notes
- `SECURITY.md` and explicit crate safety notes for the public release

### Notes

- The Rust implementation is production-oriented `f64` code built against a
  Lean formal specification.
- This release does not yet include a formal floating-point refinement proof
  from Rust `f64` execution to the Lean `Real` model.
