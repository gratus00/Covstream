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