/// Output layout used when extracting covariance or shrinkage matrices.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MatrixLayout {
    /// Full dense matrix in row-major order.
    RowMajor,
    /// Symmetric matrix stored as its packed upper triangle.
    ///
    /// The entry order is `(0,0), (0,1), ..., (0,k-1), (1,1), (1,2), ...`.
    UpperTrianglePacked,
}

impl MatrixLayout {
    /// Returns the minimum buffer size required for this layout at dimension
    /// `dimension`.
    pub fn output_size(self, dimension: usize) -> usize {
        match self {
            MatrixLayout::RowMajor => dimension * dimension,
            MatrixLayout::UpperTrianglePacked => dimension * (dimension + 1) / 2,
        }
    }
}
