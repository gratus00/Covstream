#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShrinkageMode{
    FixedAlpha(f64),
    ClippedAlpha(f64),
}

impl ShrinkageMode{
    pub fn alpha(self)->f64{
        match self{
            ShrinkageMode::FixedAlpha(alpha)=>alpha,
            ShrinkageMode::ClippedAlpha(alpha)=>alpha.clamp(0.0, 1.0),
        }
    }
}

pub fn diagonal_mean_row_major(matrix: &[f64], dimension: usize) -> f64{
    let mut sum = 0.0;
    for i in 0..dimension{
        sum += matrix[i*dimension+i];
    }
    sum / dimension as f64
}

pub fn scaled_identity_row_major(mu: f64, dimension: usize) -> Vec<f64> {
    let mut out = vec![0.0; dimension * dimension];

    for i in 0..dimension {
        out[i * dimension + i] = mu;
    }

    out
}

pub fn shrink_row_major(covariance: &[f64], dimension: usize, alpha: f64) -> Vec<f64>{
    let mu = diagonal_mean_row_major(covariance, dimension);
    let target = scaled_identity_row_major(mu, dimension);
    let mut out = vec![0.0; covariance.len()];

    for i in 0..covariance.len(){
        out[i] = (1.0 - alpha) * covariance[i] + alpha * target[i];
    }

    out
}

pub fn shrink_with_mode_row_major(
    covariance: &[f64],
    dimension: usize,
    mode: ShrinkageMode,
) -> Vec<f64> {
    let alpha = mode.alpha();
    shrink_row_major(covariance, dimension, alpha)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_mean_row_major_returns_expected_value() {
        let matrix = vec![2.0, 1.0,
                          1.0, 4.0];

        let mean = diagonal_mean_row_major(&matrix, 2);

        assert_eq!(mean, 3.0);
    }

    #[test]
    fn scaled_identity_row_major_returns_expected_matrix() {
        let matrix = scaled_identity_row_major(3.0, 2);

        assert_eq!(matrix, vec![3.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn shrink_row_major_with_zero_alpha_returns_original_matrix() {
        let covariance = vec![2.0, 1.0,
                            1.0, 4.0];

        let shrunk = shrink_row_major(&covariance, 2, 0.0);

        assert_eq!(shrunk, covariance);
    }

    #[test]
    fn shrink_row_major_with_one_alpha_returns_target_matrix() {
        let covariance = vec![2.0, 1.0,
                            1.0, 4.0];

        let shrunk = shrink_row_major(&covariance, 2, 1.0);

        assert_eq!(shrunk, vec![3.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn shrink_row_major_with_half_alpha_blends_covariance_and_target() {
        let covariance = vec![2.0, 1.0,
                            1.0, 4.0];

        let shrunk = shrink_row_major(&covariance, 2, 0.5);

        assert_eq!(shrunk, vec![2.5, 0.5, 0.5, 3.5]);
    }

    #[test]
    fn shrink_with_mode_row_major_uses_fixed_alpha() {
        let covariance = vec![2.0, 1.0,
                            1.0, 4.0];

        let shrunk = shrink_with_mode_row_major(
            &covariance,
            2,
            ShrinkageMode::FixedAlpha(0.5),
        );

        assert_eq!(shrunk, vec![2.5, 0.5, 0.5, 3.5]);
    }

    #[test]
    fn shrink_with_mode_row_major_clips_large_alpha_to_one() {
        let covariance = vec![2.0, 1.0,
                            1.0, 4.0];

        let shrunk = shrink_with_mode_row_major(
            &covariance,
            2,
            ShrinkageMode::ClippedAlpha(2.0),
        );

        assert_eq!(shrunk, vec![3.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn shrink_with_mode_row_major_clips_negative_alpha_to_zero() {
        let covariance = vec![2.0, 1.0,
                            1.0, 4.0];

        let shrunk = shrink_with_mode_row_major(
            &covariance,
            2,
            ShrinkageMode::ClippedAlpha(-1.0),
        );

        assert_eq!(shrunk, covariance);
    }
}