use crate::kernels::{axpy_in_place, scale_into};
use crate::packing::packed_len;
use crate::{CovstreamError, ShrinkageMode};
use rayon::prelude::*;

const PARALLEL_MIN_WORK: usize = 1 << 20;
const TARGET_TASKS_PER_THREAD: usize = 2;
const MIN_CHUNK_SAMPLES: usize = 32;

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
}

impl CovstreamCore {
    fn new_zeroed(dimension: usize) -> Self {
        debug_assert!(dimension > 0);

        Self {
            dimension,
            sample_count: 0,
            mean: vec![0.0; dimension],
            cov_numerator: vec![0.0; packed_len(dimension)],
            scratch_delta: vec![0.0; dimension],
        }
    }

    /// Creates a new zeroed streaming state for a fixed dimension.
    ///
    /// Returns [`CovstreamError::ZeroDimension`] when `dimension == 0`.
    pub fn new(dimension: usize) -> Result<Self, CovstreamError> {
        if dimension == 0 {
            return Err(CovstreamError::ZeroDimension);
        }

        Ok(Self::new_zeroed(dimension))
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
            let delta = value - self.mean[i];
            self.scratch_delta[i] = delta;
            self.mean[i] += delta / nf;
        }

        let residual_scale = (next_count - 1) as f64 / nf;
        let mut packed = 0;

        for row in 0..dimension {
            let len = dimension - row;
            let scale = self.scratch_delta[row] * residual_scale;
            axpy_in_place(
                &mut self.cov_numerator[packed..packed + len],
                &self.scratch_delta[row..],
                scale,
            );
            packed += len;
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
        let dimension = self.dimension;
        if !samples.len().is_multiple_of(dimension) {
            return Err(CovstreamError::MalformedBatchInput {
                dimension,
                len: samples.len(),
            });
        }

        for sample in samples.chunks_exact(dimension) {
            self.observe_validated(sample);
        }

        Ok(())
    }

    fn observe_batch_row_major_validated(&mut self, samples: &[f64]) {
        let dimension = self.dimension;
        for sample in samples.chunks_exact(dimension) {
            self.observe_validated(sample);
        }
    }

    fn should_parallelize_batch(&self, sample_count: usize) -> bool {
        sample_count > 1
            && rayon::current_num_threads() > 1
            && sample_count
                .saturating_mul(self.dimension)
                .saturating_mul(self.dimension)
                >= PARALLEL_MIN_WORK
    }

    fn merge_same_dimension(&mut self, other: &Self) {
        let dimension = self.dimension;

        if other.sample_count == 0 {
            return;
        }

        if self.sample_count == 0 {
            self.sample_count = other.sample_count;
            self.mean.copy_from_slice(&other.mean);
            self.cov_numerator.copy_from_slice(&other.cov_numerator);
            self.scratch_delta.fill(0.0);
            return;
        }

        for i in 0..dimension {
            self.scratch_delta[i] = other.mean[i] - self.mean[i];
        }

        axpy_in_place(&mut self.cov_numerator, &other.cov_numerator, 1.0);

        let left = self.sample_count as f64;
        let right = other.sample_count as f64;
        let merged_count = self.sample_count + other.sample_count;
        let merged = merged_count as f64;
        let correction = left * right / merged;

        let mut packed = 0;
        for row in 0..dimension {
            let len = dimension - row;
            let scale = self.scratch_delta[row] * correction;
            axpy_in_place(
                &mut self.cov_numerator[packed..packed + len],
                &self.scratch_delta[row..],
                scale,
            );
            packed += len;
        }

        let right_weight = right / merged;
        for i in 0..dimension {
            self.mean[i] += self.scratch_delta[i] * right_weight;
        }
        self.sample_count = merged_count;
    }

    /// Merges another state of the same dimension into `self`.
    ///
    /// This is intended for parallel or sharded ingest workflows where
    /// independent partial states are accumulated separately and combined later.
    pub fn merge(&mut self, other: &Self) -> Result<(), CovstreamError> {
        let dimension = self.dimension;

        if dimension != other.dimension {
            return Err(CovstreamError::WrongDimension {
                expected: dimension,
                got: other.dimension,
            });
        }

        self.merge_same_dimension(other);
        Ok(())
    }

    /// Ingests a flat row-major batch using parallel partial-state reduction
    /// when the batch is large enough to justify it.
    ///
    /// Small batches still use the serial path to avoid Rayon overhead.
    pub fn observe_batch_row_major_parallel(
        &mut self,
        samples: &[f64],
    ) -> Result<(), CovstreamError> {
        self.check_batch_row_major(samples)?;
        self.observe_batch_row_major_parallel_trusted_finite(samples)
    }

    /// Parallel batch ingest variant that trusts the caller that all values are
    /// finite while still enforcing batch shape checks.
    pub fn observe_batch_row_major_parallel_trusted_finite(
        &mut self,
        samples: &[f64],
    ) -> Result<(), CovstreamError> {
        let dimension = self.dimension;

        if !samples.len().is_multiple_of(dimension) {
            return Err(CovstreamError::MalformedBatchInput {
                dimension,
                len: samples.len(),
            });
        }

        let sample_count = samples.len() / dimension;
        if !self.should_parallelize_batch(sample_count) {
            self.observe_batch_row_major_validated(samples);
            return Ok(());
        }

        let target_tasks = (rayon::current_num_threads() * TARGET_TASKS_PER_THREAD)
        .min(sample_count.div_ceil(MIN_CHUNK_SAMPLES).max(1))
        .min(sample_count);

        let chunk_samples = sample_count.div_ceil(target_tasks);

        let merged = samples
            .par_chunks(chunk_samples * dimension)
            .map(|chunk| {
                let mut partial = CovstreamCore::new_zeroed(dimension);
                partial.observe_batch_row_major_validated(chunk);
                partial
            })
            .reduce(
                || CovstreamCore::new_zeroed(dimension),
                |mut left, right| {
                    left.merge_same_dimension(&right);
                    left
                },
            );

        self.merge(&merged)
    }

    /// Resets the state to its initial zeroed form without reallocating buffers.
    pub fn reset(&mut self) {
        self.sample_count = 0;
        self.mean.fill(0.0);
        self.cov_numerator.fill(0.0);
        self.scratch_delta.fill(0.0);
    }

    fn covariance_denominator(&self) -> Result<f64, CovstreamError> {
        if self.sample_count < 2 {
            return Err(CovstreamError::InsufficientSamples {
                actual: self.sample_count as usize,
            });
        }

        Ok((self.sample_count - 1) as f64)
    }

    fn diagonal_mean_from_covariance(&self, inv_denominator: f64) -> f64 {
        let mut diagonal_sum = 0.0;
        let mut packed = 0;
        let dimension = self.dimension;

        for row in 0..dimension {
            diagonal_sum += self.cov_numerator[packed];
            packed += dimension - row;
        }

        diagonal_sum * inv_denominator / dimension as f64
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
        let inv_denominator = 1.0 / self.covariance_denominator()?;
        let mut packed = 0;

        for row in 0..dimension {
            let row_offset = row * dimension;

            for col in row..dimension {
                let value = self.cov_numerator[packed] * inv_denominator;
                out[row_offset + col] = value;
                if row != col {
                    out[col * dimension + row] = value;
                }
                packed += 1;
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

        let inv_denominator = 1.0 / self.covariance_denominator()?;
        scale_into(&mut out[..expected], &self.cov_numerator, inv_denominator);
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
        let inv_denominator = 1.0 / self.covariance_denominator()?;
        let mu = self.diagonal_mean_from_covariance(inv_denominator);

        let offdiag_scale = (1.0 - alpha) * inv_denominator;
        let diag_scale = offdiag_scale;
        let diag_bias = alpha * mu;

        let mut packed = 0;

        for row in 0..dimension {
            let row_offset = row * dimension;
            for col in row..dimension {
                let base = self.cov_numerator[packed];
                let value = if row == col {
                    base * diag_scale + diag_bias
                } else {
                    base * offdiag_scale
                };

                out[row_offset + col] = value;
                if row != col {
                    out[col * dimension + row] = value;
                }

                packed += 1;
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
        let inv_denominator = 1.0 / self.covariance_denominator()?;
        let mu = self.diagonal_mean_from_covariance(inv_denominator);
        let offdiag_scale = (1.0 - alpha) * inv_denominator;
        let diag_bias = alpha * mu;

        let mut packed = 0;
        for row in 0..dimension {
            let len = dimension - row;
            scale_into(
                &mut out[packed..packed + len],
                &self.cov_numerator[packed..packed + len],
                offdiag_scale,
            );
            out[packed] += diag_bias;
            packed += len;
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
