#[derive(Debug, Clone, PartialEq)]
pub enum CovstreamError {
    ZeroDimension,
    WrongDimension { expected: usize, got: usize },
    InsufficientSamples { actual: usize },
    NonFiniteInput,
    OutputBufferTooSmall { expected: usize, got: usize },
}