use crate::{CovstreamError, MatrixLayout};

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

    pub fn covariance_row_major(&self)->Result<Vec<f64>, CovstreamError>{
        if self.sample_count < 2 {
            return Err(CovstreamError::InsufficientSamples { 
                actual: (self.sample_count as usize), 
            });
        }

        let dimension = self.dimension;
        let denominator = (self.sample_count - 1) as f64;
        let mut out = vec![0.0; dimension * dimension];

        for row in 0..dimension{
            for col in 0..dimension{
                let packed = if row <= col {
                    packed_index(dimension, row, col)
                } else {
                    packed_index(dimension, col, row)
                };
                out[row * dimension + col] = self.cov_numerator[packed]/denominator;
            }
        }
        Ok(out)
    }

    pub fn covariance_upper_triangle_packed(&self)-> Result<Vec<f64>, CovstreamError>{
        if self.sample_count < 2{
            return Err(CovstreamError::InsufficientSamples { 
                actual: self.sample_count as usize,
            });
        }

        let denominator = (self.sample_count - 1) as f64;
        let mut out = vec![0.0; self.cov_numerator.len()];

        for i in 0..self.cov_numerator.len(){
            out[i]=self.cov_numerator[i]/denominator;
        }

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

    pub fn covarience_buffer(&self)->Result<Vec<f64>, CovstreamError>{
        match self.layout{
            MatrixLayout::RowMajor => self.core.covariance_row_major(),
            MatrixLayout::UpperTrianglePacked => self.core.covariance_upper_triangle_packed(),
        }
    }

    pub fn observe(&mut self, sample: &[f64]) -> Result<(), CovstreamError> {
        self.core.observe(sample)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_new_rejects_zero_dimension() {
        let result = CovstreamCore::new(0);

        match result {
            Err(CovstreamError::ZeroDimension) => {}
            Err(other) => panic!("unexpected error: {:?}", other),
            Ok(_) => panic!("expected ZeroDimension error, got success"),
        }
    }

    #[test]
    fn core_new_initializes_zero_state() {
        let result = CovstreamCore::new(3);

        match result {
            Ok(state) => {
                assert_eq!(state.dimension(), 3);
                assert_eq!(state.sample_count(), 0);
            }
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }

    #[test]
    fn state_new_uses_requested_layout() {
        let result = CovstreamState::new(3, MatrixLayout::RowMajor);

        match result {
            Ok(state) => {
                assert_eq!(state.layout(), MatrixLayout::RowMajor);
                assert_eq!(state.dimension(), 3);
                assert_eq!(state.sample_count(), 0);
            }
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }

    #[test]
    fn observe_rejects_wrong_dimension() {
        let state_result = CovstreamCore::new(2);

        let mut state = match state_result {
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        let result = state.observe(&[1.0]);

        match result {
            Err(CovstreamError::WrongDimension { expected, got }) => {
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            Err(other) => panic!("unexpected error: {:?}", other),
            Ok(()) => panic!("expected WrongDimension error, got success"),
        }
    }
    #[test]
    fn observe_rejects_non_finite_input() {
        let state_result = CovstreamCore::new(2);

        let mut state = match state_result {
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        let result = state.observe(&[1.0, f64::NAN]);

        match result {
            Err(CovstreamError::NonFiniteInput) => {}
            Err(other) => panic!("unexpected error: {:?}", other),
            Ok(()) => panic!("expected NonFiniteInput error, got success"),
        }
    }

    #[test]
    fn observe_increments_sample_count() {
        let state_result = CovstreamCore::new(2);

        let mut state = match state_result {
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        let result = state.observe(&[1.0, 2.0]);

        match result {
            Ok(()) => {
                assert_eq!(state.sample_count(), 1);
            }
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }

    #[test]
    fn observe_updates_mean(){
        let result = CovstreamCore::new(2);
        let mut state = match result{
            Ok(state)=> state,
            Err(err)=> panic!("expected success, got error: {:?}", err),
        };

        let first = state.observe(&[2.0, 4.0]);
        match first {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        assert_eq!(state.mean(), &[2.0, 4.0]);

        let second = state.observe(&[4.0, 8.0]);
        match second {
            Ok(()) => {}
            Err(err)=> panic!("expected success, got error: {:?}", err),
        }

        assert_eq!(state.mean(), &[3.0, 6.0]);
    }

    #[test]
    fn observe_updates_covariance_numerator(){
        let result = CovstreamCore::new(2);
        let mut state = match result{
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        match state.observe(&[1.0, 2.0]){
            Ok(()) => {}
            Err(err)=> panic!("expected success, got error: {:?}", err),
        }

        assert_eq!(state.cov_numerator(), &[0.0, 0.0, 0.0]);

        match state.observe(&[3.0, 4.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        assert_eq!(state.cov_numerator(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn covariance_row_major_rejects_insufficient_samples() {
        let result = CovstreamCore::new(2);

        let state = match result {
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        let covariance = state.covariance_row_major();

        match covariance {
            Err(CovstreamError::InsufficientSamples { actual }) => {
                assert_eq!(actual, 0);
            }
            Err(other) => panic!("unexpected error: {:?}", other),
            Ok(_) => panic!("expected InsufficientSamples error, got success"),
        }
    }

    #[test]
    fn covariance_row_major_returns_expected_2x2_matrix() {
        let result = CovstreamCore::new(2);

        let mut state = match result {
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        match state.observe(&[1.0, 2.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        match state.observe(&[3.0, 4.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        let covariance = state.covariance_row_major();

        match covariance {
            Ok(matrix) => {
                assert_eq!(matrix, vec![2.0, 2.0, 2.0, 2.0]);
            }
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }

    #[test]
    fn covariance_upper_triangle_packed_returns_expected_2x2_output() {
        let result = CovstreamCore::new(2);

        let mut state = match result {
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        match state.observe(&[1.0, 2.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        match state.observe(&[3.0, 4.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        let covariance = state.covariance_upper_triangle_packed();

        match covariance {
            Ok(matrix) => {
                assert_eq!(matrix, vec![2.0, 2.0, 2.0]);
            }
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }

    #[test]
    fn state_covariance_buffer_uses_row_major_layout() {
        let result = CovstreamState::new(2, MatrixLayout::RowMajor);

        let mut state = match result {
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        match state.observe(&[1.0, 2.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        match state.observe(&[3.0, 4.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        let covariance = state.covarience_buffer();

        match covariance {
            Ok(matrix) => {
                assert_eq!(matrix, vec![2.0, 2.0, 2.0, 2.0]);
            }
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }

    #[test]
    fn state_covariance_buffer_uses_packed_layout() {
        let result = CovstreamState::new(2, MatrixLayout::UpperTrianglePacked);

        let mut state = match result {
            Ok(state) => state,
            Err(err) => panic!("expected success, got error: {:?}", err),
        };

        match state.observe(&[1.0, 2.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        match state.observe(&[3.0, 4.0]) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }

        let covariance = state.covarience_buffer();

        match covariance {
            Ok(matrix) => {
                assert_eq!(matrix, vec![2.0, 2.0, 2.0]);
            }
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }
}
