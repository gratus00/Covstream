use std::fmt;

/// Runtime errors returned by the checked Covstream APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum CovstreamError {
    /// The requested dimension was zero.
    ZeroDimension,
    /// One sample had the wrong length.
    WrongDimension { expected: usize, got: usize },
    /// A flat batch buffer was not a whole number of samples.
    MalformedBatchInput { dimension: usize, len: usize },
    /// Covariance or shrinkage was requested before two samples were observed.
    InsufficientSamples { actual: usize },
    /// A checked ingest path encountered `NaN` or `Inf`.
    NonFiniteInput,
    /// The caller-provided output buffer was too small for the requested layout.
    OutputBufferTooSmall { expected: usize, got: usize },
}

impl fmt::Display for CovstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CovstreamError::ZeroDimension => write!(f, "dimension must be greater than zero"),
            CovstreamError::WrongDimension { expected, got } => {
                write!(
                    f,
                    "wrong sample dimension: expected {}, got {}",
                    expected, got
                )
            }
            CovstreamError::MalformedBatchInput { dimension, len } => {
                write!(
                    f,
                    "batch input length must be a multiple of dimension: dimension {}, len {}",
                    dimension, len
                )
            }
            CovstreamError::InsufficientSamples { actual } => {
                write!(f, "insufficient samples: need at least 2, got {}", actual)
            }
            CovstreamError::NonFiniteInput => write!(f, "sample contains NaN or infinity"),
            CovstreamError::OutputBufferTooSmall { expected, got } => {
                write!(
                    f,
                    "output buffer too small: expected {}, got {}",
                    expected, got
                )
            }
        }
    }
}

impl std::error::Error for CovstreamError {}
