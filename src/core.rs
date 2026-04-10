use crate::{CovstreamError, MatrixLayout, ShrinkageMode};
use crate::shrinkage::shrink_with_mode_row_major;

#[derive(Debug, Clone)]
pub struct CovstreamCore{
    dimension: usize,
    sample_count:u64,
    mean:Vec<f64>,
    cov_numerator:Vec<f64>,
    scratch_delta:Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CovstreamState{
    layout: MatrixLayout,
    core: CovstreamCore,
}

impl CovstreamCore{
    pub fn new(dimension:usize)-> Result<Self, CovstreamError>{
        if dimension==0{
            return Err(CovstreamError::ZeroDimension);
        }

        Ok(Self {
            dimension,
            sample_count:0,
            mean: vec![0.0; dimension],
            cov_numerator: vec![0.0; packed_len(dimension)],
            scratch_delta: vec![0.0; dimension],
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn cov_numerator(&self) -> &[f64] {
        &self.cov_numerator
    }

    pub fn check_sample(&self, sample:&[f64])->Result<(), CovstreamError>{
        if sample.len()!= self.dimension{
            return Err(CovstreamError::WrongDimension {
                expected: self.dimension,
                got: sample.len(), 
            });
        }

        if sample.iter().any(|x| !x.is_finite()){
            return Err(CovstreamError::NonFiniteInput);
        }

        Ok(())
    }

    pub fn observe(&mut self, sample: &[f64])->Result<(), CovstreamError>{
        self.check_sample(sample)?;
        let next_count = self.sample_count+1;
        let nf = next_count as f64;

        for(i, value) in sample.iter().copied().enumerate() {
            self.scratch_delta[i]=value-self.mean[i];
        }

        for i in 0..self.dimension { 
            self.mean[i]+=self.scratch_delta[i]/nf;
        }

        for i in 0..self.dimension {
            for j in i..self.dimension{
                let idx = packed_index(self.dimension, i, j);
                self.cov_numerator[idx] += self.scratch_delta[i] * (sample[j] - self.mean[j]);
            }
        }

        self.sample_count=next_count;
        Ok(()) 
    }

    pub fn covariance_row_major_into(&self, out: &mut [f64]) -> Result<(), CovstreamError>{
        if self.sample_count < 2 {
            return Err(CovstreamError::InsufficientSamples { 
                actual: (self.sample_count as usize), 
            });
        }

        let expected = self.dimension * self.dimension;
        if out.len() < expected { 
            return Err(CovstreamError::OutputBufferTooSmall { 
                expected,
                got: out.len(),
            });
        }

        let denominator = (self.sample_count - 1) as f64;

        for row in 0..self.dimension {
            for col in 0..self.dimension {
                let packed = if row <= col {
                    packed_index(self.dimension, row, col)
                } else {
                    packed_index(self.dimension, col, row)
                };

                out[row*self.dimension + col] = self.cov_numerator[packed] / denominator;
            }
        }

        Ok(())
    }

    pub fn covariance_row_major(&self)->Result<Vec<f64>, CovstreamError>{
        let mut out = vec![0.0; self.dimension * self.dimension];
        self.covariance_row_major_into(&mut out)?;
        Ok(out)
    }

    pub fn covariance_upper_triangle_packed_into(&self, out: &mut [f64]) -> Result<(), CovstreamError> {
        if self.sample_count < 2{
            return Err(CovstreamError::InsufficientSamples { 
                actual: self.sample_count as usize,
            });
        }

        let expected = self.cov_numerator.len();
        if out.len() < expected {
            return Err(CovstreamError::OutputBufferTooSmall { 
                expected, 
                got: out.len(),
            });
        }

        let denominator = (self.sample_count - 1) as f64;
        for i in 0..self.cov_numerator.len(){
            out[i] = self.cov_numerator[i] / denominator;
        }

        Ok(())
    }

    pub fn covariance_upper_triangle_packed(&self)-> Result<Vec<f64>, CovstreamError>{
        let mut out = vec![0.0; self.cov_numerator.len()];
        self.covariance_upper_triangle_packed_into(&mut out)?;
        Ok(out)
    }

    pub fn ledoit_wolf_row_major_into(&self, mode:ShrinkageMode, out: &mut [f64]) -> Result<(), CovstreamError> {
        let covariance = self.covariance_row_major()?;
        let expected = self.dimension * self.dimension;

        if out.len() < expected { 
            return Err(CovstreamError::OutputBufferTooSmall { 
                expected,
                got: out.len(),
            });
        }

        let shrunk = shrink_with_mode_row_major(&covariance, self.dimension, mode);
        for i in 0..expected {
            out[i] = shrunk[i];
        }

        Ok(())
    }

    pub fn ledoit_wolf_row_major(&self, mode:ShrinkageMode) -> Result<Vec<f64>, CovstreamError>{
        let mut out  = vec![0.0; self.dimension * self.dimension];
        self.ledoit_wolf_row_major_into(mode, &mut out)?;
        Ok(out)
    }
}

impl CovstreamState {
    pub fn new(dimension: usize, layout: MatrixLayout) -> Result<Self, CovstreamError> {
        Ok(Self {
            layout,
            core: CovstreamCore::new(dimension)?,
        })
    }

    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    pub fn dimension(&self) -> usize {
        self.core.dimension()
    }

    pub fn sample_count(&self) -> u64 {
        self.core.sample_count()
    }

    pub fn covariance_row_major(&self)->Result<Vec<f64>, CovstreamError>{
        self.core.covariance_row_major()
    }

    pub fn covariance_buffer_into(&self, out: &mut [f64]) -> Result<(), CovstreamError> {
        match self.layout {
            MatrixLayout::RowMajor => self.core.covariance_row_major_into(out),
            MatrixLayout::UpperTrianglePacked => self.core.covariance_upper_triangle_packed_into(out),
        }
    }

    pub fn covariance_buffer(&self)->Result<Vec<f64>, CovstreamError>{
        match self.layout{
            MatrixLayout::RowMajor => self.core.covariance_row_major(),
            MatrixLayout::UpperTrianglePacked => self.core.covariance_upper_triangle_packed(),
        }
    }

    pub fn observe(&mut self, sample: &[f64]) -> Result<(), CovstreamError> {
        self.core.observe(sample)
    }

    pub fn ledoit_wolf_row_major(&self, mode: ShrinkageMode)->Result<Vec<f64>, CovstreamError>{
        self.core.ledoit_wolf_row_major(mode)
    }

    pub fn ledoit_wolf_buffer_into(
        &self,
        mode: ShrinkageMode,
        out: &mut [f64],
    ) -> Result<(), CovstreamError> {
        match self.layout {
            MatrixLayout::RowMajor => self.core.ledoit_wolf_row_major_into(mode, out),
            MatrixLayout::UpperTrianglePacked => {
                let expected = packed_len(self.dimension());
                if out.len() < expected {
                    return Err(CovstreamError::OutputBufferTooSmall {
                        expected,
                        got: out.len(),
                    });
                }

                let row_major = self.core.ledoit_wolf_row_major(mode)?;
                for row in 0..self.dimension() {
                    for col in row..self.dimension() {
                        let packed = packed_index(self.dimension(), row, col);
                        out[packed] = row_major[row * self.dimension() + col];
                    }
                }

                Ok(())
            }
        }
    }

    pub fn ledoit_wolf_buffer(&self, mode: ShrinkageMode) -> Result<Vec<f64>, CovstreamError> {
        match self.layout {
            MatrixLayout::RowMajor => self.core.ledoit_wolf_row_major(mode),
            MatrixLayout::UpperTrianglePacked => {
                let row_major = self.core.ledoit_wolf_row_major(mode)?;
                Ok(row_major_to_upper_triangle_packed(&row_major, self.dimension()))
            }
        }
    } 
}


fn packed_len(dimension: usize) -> usize {
    dimension * (dimension + 1) / 2
}

fn packed_index(dimension: usize, row: usize, col: usize) -> usize {
    debug_assert!(row<=col);
    debug_assert!(col<dimension);

    row * dimension - (row * row.saturating_sub(1))/2 + (col - row)
}

fn row_major_to_upper_triangle_packed(matrix: &[f64], dimension: usize) -> Vec<f64> {
    let mut out = vec![0.0; packed_len(dimension)];

    for row in 0..dimension { 
        for col in row..dimension {
            let packed = packed_index(dimension, row, col);
            out[packed] = matrix[row*dimension + col];
        }
    }
    out
}
