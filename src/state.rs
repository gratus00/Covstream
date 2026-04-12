use crate::core::CovstreamCore;
use crate::{CovstreamError, MatrixLayout, ShrinkageMode};

/// User-facing wrapper around [`CovstreamCore`] that remembers an output layout.
///
/// This is the default entry point for library consumers who want a simple API:
///
/// - ingest samples
/// - choose a preferred layout once
/// - extract covariance or shrinkage buffers in that layout
#[derive(Debug, Clone)]
pub struct CovstreamState {
    layout: MatrixLayout,
    core: CovstreamCore,
}

impl CovstreamState {
    /// Creates a new fixed-dimension state with a chosen default output layout.
    pub fn new(dimension: usize, layout: MatrixLayout) -> Result<Self, CovstreamError> {
        Ok(Self {
            layout,
            core: CovstreamCore::new(dimension)?,
        })
    }

    /// Returns the layout used by [`Self::covariance_buffer`] and
    /// [`Self::ledoit_wolf_buffer`].
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    /// Returns the fixed sample dimension.
    pub fn dimension(&self) -> usize {
        self.core.dimension()
    }

    /// Returns the number of observed samples.
    pub fn sample_count(&self) -> u64 {
        self.core.sample_count()
    }

    /// Returns the covariance matrix in row-major layout regardless of the
    /// state's default layout.
    pub fn covariance_row_major(&self) -> Result<Vec<f64>, CovstreamError> {
        self.core.covariance_row_major()
    }

    /// Writes covariance output into a caller-provided buffer using the state's
    /// default layout.
    pub fn covariance_buffer_into(&self, out: &mut [f64]) -> Result<(), CovstreamError> {
        match self.layout {
            MatrixLayout::RowMajor => self.core.covariance_row_major_into(out),
            MatrixLayout::UpperTrianglePacked => {
                self.core.covariance_upper_triangle_packed_into(out)
            }
        }
    }

    /// Allocates and returns covariance output using the state's default layout.
    pub fn covariance_buffer(&self) -> Result<Vec<f64>, CovstreamError> {
        let mut out = vec![0.0; self.layout.output_size(self.dimension())];
        self.covariance_buffer_into(&mut out)?;
        Ok(out)
    }

    /// Ingests one sample through the safe checked path.
    pub fn observe(&mut self, sample: &[f64]) -> Result<(), CovstreamError> {
        self.core.observe(sample)
    }

    /// Ingests a flat row-major batch through the safe checked path.
    ///
    /// The slice contains back-to-back samples of length [`Self::dimension`].
    pub fn observe_batch_row_major(&mut self, samples: &[f64]) -> Result<(), CovstreamError> {
        self.core.observe_batch_row_major(samples)
    }

    /// Ingests one sample while trusting the caller that all values are finite.
    pub fn observe_trusted_finite(&mut self, sample: &[f64]) -> Result<(), CovstreamError> {
        self.core.observe_trusted_finite(sample)
    }

    /// Ingests a flat row-major batch while trusting the caller that all values
    /// are finite.
    pub fn observe_batch_row_major_trusted_finite(
        &mut self,
        samples: &[f64],
    ) -> Result<(), CovstreamError> {
        self.core.observe_batch_row_major_trusted_finite(samples)
    }

    /// Resets the wrapped core state without reallocating buffers.
    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// Returns the shrunk covariance matrix in row-major layout regardless of
    /// the state's default layout.
    pub fn ledoit_wolf_row_major(&self, mode: ShrinkageMode) -> Result<Vec<f64>, CovstreamError> {
        self.core.ledoit_wolf_row_major(mode)
    }

    /// Writes shrunk covariance output into a caller-provided buffer using the
    /// state's default layout.
    pub fn ledoit_wolf_buffer_into(
        &self,
        mode: ShrinkageMode,
        out: &mut [f64],
    ) -> Result<(), CovstreamError> {
        match self.layout {
            MatrixLayout::RowMajor => self.core.ledoit_wolf_row_major_into(mode, out),
            MatrixLayout::UpperTrianglePacked => {
                self.core.ledoit_wolf_upper_triangle_packed_into(mode, out)
            }
        }
    }

    /// Allocates and returns shrunk covariance output using the state's default
    /// layout.
    pub fn ledoit_wolf_buffer(&self, mode: ShrinkageMode) -> Result<Vec<f64>, CovstreamError> {
        let mut out = vec![0.0; self.layout.output_size(self.dimension())];
        self.ledoit_wolf_buffer_into(mode, &mut out)?;
        Ok(out)
    }

    /// Merges another state with the same dimension into `self`.
    ///
    /// This is useful when independent partial states are accumulated in
    /// separate tasks and combined later.
    pub fn merge(&mut self, other: &Self) -> Result<(), CovstreamError> {
        self.core.merge(&other.core)
    }

    /// Ingests a flat row-major batch and uses the parallel reduction path when
    /// the batch is large enough to benefit from it.
    pub fn observe_batch_row_major_parallel(
        &mut self,
        samples: &[f64],
    ) -> Result<(), CovstreamError> {
        self.core.observe_batch_row_major_parallel(samples)
    }

    /// Parallel batch ingest variant that trusts the caller that all values are
    /// finite while still enforcing batch shape checks.
    pub fn observe_batch_row_major_parallel_trusted_finite(
        &mut self,
        samples: &[f64],
    ) -> Result<(), CovstreamError> {
        self.core
            .observe_batch_row_major_parallel_trusted_finite(samples)
    }
}
