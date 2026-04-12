# Launch Notes

This file keeps the public-facing copy close to the code that ships.

## One-Line Description

Lean-backed Rust library for fixed-dimension streaming covariance and
Ledoit-Wolf shrinkage.

## Slightly Longer Description

`covstream` lets developers maintain a covariance estimate incrementally as new
aligned vectors arrive, then extract either dense or packed symmetric outputs on
demand. The repository pairs a production-oriented Rust `f64` implementation
with a Lean formal specification of the mathematical contract.

## GitHub Repo Description

Lean-backed Rust library for fixed-dimension streaming covariance and
Ledoit-Wolf shrinkage.

## Suggested GitHub Topics

- `rust`
- `lean4`
- `mathlib`
- `covariance`
- `statistics`
- `quant`
- `risk`
- `formal-methods`

## Short Launch Post

I just open-sourced `covstream`: a Lean-backed Rust library for
fixed-dimension streaming covariance and Ledoit-Wolf shrinkage.

It is built for cases where you want to keep covariance updated as new aligned
vectors arrive, without rebuilding from scratch every time. The crate supports
safe and trusted-finite ingest, packed symmetric output, reusable buffers,
examples, benchmarks, CI, and a Lean specification in the same repo.

Repo: https://github.com/gratus00/Covstream

## Longer Launch Post

I just published `covstream`, a small Rust library for fixed-dimension
streaming covariance and Ledoit-Wolf shrinkage.

The main use case is quant or analytics pipelines where new aligned vectors
arrive over time and you want to keep the covariance estimate updated
incrementally instead of rebuilding it from scratch. The crate supports
single-sample ingest, flat row-major batch ingest, packed upper-triangle output,
reusable output buffers, and optional parallel batch reduction for larger
offline jobs.

What makes the project interesting to me is that the Rust implementation sits
next to a Lean formalization of the mathematical contract. It is not claiming a
full floating-point refinement proof yet, but the repo keeps the math and the
production-oriented implementation tied together in one place.

Repo: https://github.com/gratus00/Covstream

## Release Checklist

- confirm `git status` is clean except for intentional release changes
- run `cargo fmt --check`
- run `cargo clippy --all-targets -- -D warnings`
- run `cargo test --all-targets`
- run `cargo package --list`
- run `cargo publish --dry-run`
- tag `v0.1.0`
- publish the crate
