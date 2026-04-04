//! Rust implementation scaffold for the `Covstream` Lean contract.
//!
//! This starts intentionally small:
//! - boundary types that mirror the Lean contract
//! - a checked constructor
//! - a checked update path
//!
//! More numerical functionality will be added incrementally.


mod error;
mod layout;


pub use error::CovstreamError;
pub use layout::MatrixLayout;



