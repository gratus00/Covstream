//! Rust implementation scaffold for the `Covstream` Lean contract.
//!
//! This starts intentionally small:
//! - boundary types that mirror the Lean contract
//! - a checked constructor
//! - a checked update path
//!
//! More numerical functionality will be added incrementally.


mod core;
mod error;
mod layout;
mod shrinkage;

pub use core::{CovstreamCore, CovstreamState};
pub use error::CovstreamError;
pub use layout::MatrixLayout;
pub use shrinkage::ShrinkageMode;



