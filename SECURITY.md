# Security Policy

## Scope

`covstream` is a numerical library for fixed-dimension streaming covariance and
shrinkage.

The crate's library API does not perform:

- network I/O
- file I/O
- subprocess execution
- credential handling
- authentication or authorization

The main security-relevant surface is memory safety and numerical robustness.

## Current Security Posture

- Checked ingest paths validate dimensions and reject `NaN` / `Inf`.
- The dependency tree is intentionally small.
- The only `unsafe` code is isolated to AArch64 SIMD leaf kernels in
  `src/kernels.rs`.

## Reporting

If you believe you found a security issue, please open a private report through
GitHub security advisories if available for the repository, or contact the
maintainer directly before opening a public issue.

Please include:

- affected version
- target platform
- minimal reproduction
- whether the issue is memory-safety, denial-of-service, data exposure, or
  numerical-integrity related

## Out of Scope

The following are generally out of scope for this crate itself:

- vulnerabilities in downstream applications that misuse the API
- unvalidated `trusted_finite` inputs supplied by callers
- operational issues in surrounding ingestion/storage/network systems
