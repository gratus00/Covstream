use crate::packing::{packed_index, packed_len};
use crate::shrinkage::shrink_entry;
use crate::{CovstreamError, ShrinkageMode};

/// Low-level fixed-dimension streaming covariance engine.
///
/// This type owns the running Welford state:
///
/// - sample count
/// - running mean vector
/// - packed upper-triangle covariance numerator
///
/// Use [`CovstreamCore`] when you want explicit control over ingest and output
/// layout. If you want a simpler API that remembers a preferred layout, use
/// [`crate::CovstreamState`] instead.
#[derive(Debug, Clone)]
pub struct CovstreamCore {
    dimension: usize,
    sample_count: u64,
    mean: Vec<f64>,
    cov_numerator: Vec<f64>,
    scratch_delta: Vec<f64>,
    scratch_residual: Vec<f64>,
}

impl CovstreamCore {
    /// Creates a new zeroed streaming state for a fixed dimension.
    ///
    /// Returns [`CovstreamError::ZeroDimension`] when `dimension == 0`.
    pub fn new(dimension: usize) -> Result<Self, CovstreamError> {
        if dimension == 0 {
            return Err(CovstreamError::ZeroDimension);
        }

        Ok(Self {
            dimension,
            sample_count: 0,
            mean: vec![0.0; dimension],
            cov_numerator: vec![0.0; packed_len(dimension)],
            scratch_delta: vec![0.0; dimension],
            scratch_residual: vec![0.0; dimension],
        })
    }

    /// Returns the fixed dimension of every sample accepted by this state.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of observed samples.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Returns the current running mean vector.
    ///
    /// The slice length is always [`Self::dimension`].
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Returns the packed upper-triangle covariance numerator.
    ///
    /// This is the internal Welford accumulator before division by `n - 1`.
    pub fn cov_numerator(&self) -> &[f64] {
        &self.cov_numerator
    }

    /// Validates one sample for safe ingest.
    ///
    /// The sample must:
    ///
    /// - have length [`Self::dimension`]
    /// - contain only finite values
    pub fn check_sample(&self, sample: &[f64]) -> Result<(), CovstreamError> {
        if sample.len() != self.dimension {
            return Err(CovstreamError::WrongDimension {
                expected: self.dimension,
                got: sample.len(),
            });
        }

        if sample.iter().any(|x| !x.is_finite()) {
            return Err(CovstreamError::NonFiniteInput);
        }

        Ok(())
    }

    fn check_batch_row_major(&self, samples: &[f64]) -> Result<(), CovstreamError> {
        if !samples.len().is_multiple_of(self.dimension) {
            return Err(CovstreamError::MalformedBatchInput {
                dimension: self.dimension,
                len: samples.len(),
            });
        }

        if samples.iter().any(|x| !x.is_finite()) {
            return Err(CovstreamError::NonFiniteInput);
        }

        Ok(())
    }

    #[inline]
    fn observe_validated(&mut self, sample: &[f64]) {
        let dimension = self.dimension;
        let next_count = self.sample_count + 1;
        let nf = next_count as f64;

        for (i, value) in sample.iter().copied().enumerate() {
            self.scratch_delta[i] = value - self.mean[i];
        }

        for i in 0..dimension {
            self.mean[i] += self.scratch_delta[i] / nf;
        }

        for (i, value) in sample.iter().copied().enumerate().take(dimension) {
            self.scratch_residual[i] = value - self.mean[i];
        }

        let mut packed = 0;

        for row in 0..dimension {
            let delta_row = self.scratch_delta[row];
            for col in row..dimension {
                self.cov_numerator[packed] += delta_row * self.scratch_residual[col];
                packed += 1;
            }
        }

        self.sample_count = next_count;
    }

    /// Ingests one sample through the safe checked path.
    ///
    /// Use this when the input may come from an external system and still needs
    /// shape and finite-value validation.
    pub fn observe(&mut self, sample: &[f64]) -> Result<(), CovstreamError> {
        self.check_sample(sample)?;
        self.observe_validated(sample);
        Ok(())
    }

    /// Ingests a flat row-major batch of back-to-back samples through the safe
    /// checked path.
    ///
    /// For a dimension `k`, the buffer must look like:
    ///
    /// `sample_0[0], ..., sample_0[k-1], sample_1[0], ..., sample_1[k-1], ...`
    ///
    /// The total slice length must be a multiple of [`Self::dimension`].
    pub fn observe_batch_row_major(&mut self, samples: &[f64]) -> Result<(), CovstreamError> {
        self.check_batch_row_major(samples)?;

        for sample in samples.chunks_exact(self.dimension) {
            self.observe_validated(sample);
        }

        Ok(())
    }

    /// Ingests one sample while trusting the caller that all values are finite.
    ///
    /// This method still enforces shape checks, but it skips `NaN` / `Inf`
    /// validation. It is intended for internal high-throughput pipelines whose
    /// upstream layer has already validated numerical values.
    pub fn observe_trusted_finite(&mut self, sample: &[f64]) -> Result<(), CovstreamError> {
        if sample.len() != self.dimension {
            return Err(CovstreamError::WrongDimension {
                expected: self.dimension,
                got: sample.len(),
            });
        }

        self.observe_validated(sample);
        Ok(())
    }

    /// Ingests a flat row-major batch while trusting the caller that all values
    /// are finite.
    ///
    /// This preserves batch-shape validation and skips the finite-value scan.
    pub fn observe_batch_row_major_trusted_finite(
        &mut self,
        samples: &[f64],
    ) -> Result<(), CovstreamError> {
        if !samples.len().is_multiple_of(self.dimension) {
            return Err(CovstreamError::MalformedBatchInput {
                dimension: self.dimension,
                len: samples.len(),
            });
        }

        for sample in samples.chunks_exact(self.dimension) {
            self.observe_validated(sample);
        }

        Ok(())
    }

    /// Resets the state to its initial zeroed form without reallocating buffers.
    pub fn reset(&mut self) {
        self.sample_count = 0;
        self.mean.fill(0.0);
        self.cov_numerator.fill(0.0);
        self.scratch_delta.fill(0.0);
        self.scratch_residual.fill(0.0);
    }

    fn covariance_denominator(&self) -> Result<f64, CovstreamError> {
        if self.sample_count < 2 {
            return Err(CovstreamError::InsufficientSamples {
                actual: self.sample_count as usize,
            });
        }

        Ok((self.sample_count - 1) as f64)
    }

    fn diagonal_mean_from_covariance(&self, denominator: f64) -> f64 {
        let mut diagonal_sum = 0.0;

        for i in 0..self.dimension {
            let packed = packed_index(self.dimension, i, i);
            diagonal_sum += self.cov_numerator[packed];
        }

        diagonal_sum / denominator / self.dimension as f64
    }

    /// Writes the sample covariance matrix into a caller-provided row-major
    /// buffer.
    ///
    /// The output slice must have length at least `dimension * dimension`.
    /// Returns [`CovstreamError::InsufficientSamples`] until at least two
    /// samples have been observed.
    pub fn covariance_row_major_into(&self, out: &mut [f64]) -> Result<(), CovstreamError> {
        let dimension = self.dimension;
        let expected = dimension * dimension;

        if out.len() < expected {
            return Err(CovstreamError::OutputBufferTooSmall {
                expected,
                got: out.len(),
            });
        }

        let denominator = self.covariance_denominator()?;

        for row in 0..dimension {
            let row_offset = row * dimension;

            for col in row..dimension {
                let packed = packed_index(dimension, row, col);
                let value = self.cov_numerator[packed] / denominator;

                out[row_offset + col] = value;

                if row != col {
                    out[col * dimension + row] = value;
                }
            }
        }

        Ok(())
    }

    /// Returns the sample covariance matrix in row-major layout.
    pub fn covariance_row_major(&self) -> Result<Vec<f64>, CovstreamError> {
        let mut out = vec![0.0; self.dimension * self.dimension];
        self.covariance_row_major_into(&mut out)?;
        Ok(out)
    }

    /// Writes the sample covariance matrix into a packed upper-triangle buffer.
    ///
    /// The layout is `(0,0), (0,1), ..., (0,k-1), (1,1), (1,2), ...`.
    pub fn covariance_upper_triangle_packed_into(
        &self,
        out: &mut [f64],
    ) -> Result<(), CovstreamError> {
        let expected = self.cov_numerator.len();

        if out.len() < expected {
            return Err(CovstreamError::OutputBufferTooSmall {
                expected,
                got: out.len(),
            });
        }

        let denominator = self.covariance_denominator()?;

        for (dst, src) in out.iter_mut().zip(self.cov_numerator.iter()) {
            *dst = *src / denominator;
        }

        Ok(())
    }

    /// Returns the sample covariance matrix in packed upper-triangle layout.
    pub fn covariance_upper_triangle_packed(&self) -> Result<Vec<f64>, CovstreamError> {
        let mut out = vec![0.0; self.cov_numerator.len()];
        self.covariance_upper_triangle_packed_into(&mut out)?;
        Ok(out)
    }

    /// Writes the shrunk covariance matrix into a caller-provided row-major
    /// buffer.
    ///
    /// Shrinkage is applied toward `μI` using the selected [`ShrinkageMode`].
    pub fn ledoit_wolf_row_major_into(
        &self,
        mode: ShrinkageMode,
        out: &mut [f64],
    ) -> Result<(), CovstreamError> {
        let dimension = self.dimension;
        let expected = dimension * dimension;

        if out.len() < expected {
            return Err(CovstreamError::OutputBufferTooSmall {
                expected,
                got: out.len(),
            });
        }

        let alpha = mode.alpha();
        let denominator = self.covariance_denominator()?;
        let mu = self.diagonal_mean_from_covariance(denominator);

        for row in 0..dimension {
            let row_offset = row * dimension;

            for col in row..dimension {
                let packed = packed_index(dimension, row, col);
                let covariance_entry = self.cov_numerator[packed] / denominator;
                let value = shrink_entry(covariance_entry, mu, row == col, alpha);

                out[row_offset + col] = value;
                if row != col {
                    out[col * dimension + row] = value;
                }
            }
        }

        Ok(())
    }

    /// Returns the shrunk covariance matrix in row-major layout.
    pub fn ledoit_wolf_row_major(&self, mode: ShrinkageMode) -> Result<Vec<f64>, CovstreamError> {
        let mut out = vec![0.0; self.dimension * self.dimension];
        self.ledoit_wolf_row_major_into(mode, &mut out)?;
        Ok(out)
    }

    /// Writes the shrunk covariance matrix into a packed upper-triangle buffer.
    pub fn ledoit_wolf_upper_triangle_packed_into(
        &self,
        mode: ShrinkageMode,
        out: &mut [f64],
    ) -> Result<(), CovstreamError> {
        let dimension = self.dimension;
        let expected = packed_len(dimension);

        if out.len() < expected {
            return Err(CovstreamError::OutputBufferTooSmall {
                expected,
                got: out.len(),
            });
        }

        let alpha = mode.alpha();
        let denominator = self.covariance_denominator()?;
        let mu = self.diagonal_mean_from_covariance(denominator);

        for row in 0..dimension {
            for col in row..dimension {
                let packed = packed_index(dimension, row, col);
                let covariance_entry = self.cov_numerator[packed] / denominator;
                out[packed] = shrink_entry(covariance_entry, mu, row == col, alpha);
            }
        }

        Ok(())
    }

    /// Returns the shrunk covariance matrix in packed upper-triangle layout.
    pub fn ledoit_wolf_upper_triangle_packed(
        &self,
        mode: ShrinkageMode,
    ) -> Result<Vec<f64>, CovstreamError> {
        let mut out = vec![0.0; packed_len(self.dimension)];
        self.ledoit_wolf_upper_triangle_packed_into(mode, &mut out)?;
        Ok(out)
    }
}
