#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MatrixLayout{
    RowMajor,
    UpperTrianglePacked,
}

impl MatrixLayout{
    pub fn output_size(self, dimension:usize) -> usize{
        match self{
            MatrixLayout::RowMajor => dimension * dimension,
            MatrixLayout::UpperTrianglePacked => dimension*(dimension+1) / 2,
        }
    }
}