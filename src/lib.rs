//! `covstream` is a fixed-dimension streaming covariance library with Lean-backed
//! specifications.
//!
//! The crate is built around two public entry points:
//!
//! - [`CovstreamState`] is the default user-facing API. It stores a chosen
//!   [`MatrixLayout`] and returns buffers in that layout.
//! - [`CovstreamCore`] is the lower-level engine. It gives direct access to the
//!   underlying state and explicit extraction methods for each layout.
//!
//! Safe ingest paths reject:
//!
//! - wrong sample dimensions
//! - malformed flat batch buffers
//! - non-finite inputs such as `NaN` and `Inf`
//!
//! Covariance and shrinkage extraction require at least two observed samples.
//! Until then the library returns [`CovstreamError::InsufficientSamples`].
//!
//! For higher-throughput pipelines that already validate inputs upstream, the
//! `trusted_finite` ingest methods skip finite-value checks while preserving
//! shape checks.

mod core;
mod error;
mod layout;
mod packing;
pub mod shrinkage;
mod state;

pub use core::CovstreamCore;
pub use error::CovstreamError;
pub use layout::MatrixLayout;
pub use shrinkage::{
    diagonal_mean_row_major, scaled_identity_row_major, shrink_row_major,
    shrink_with_mode_row_major, ShrinkageMode,
};
pub use state::CovstreamState;
