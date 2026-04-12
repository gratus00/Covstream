#[inline]
pub(crate) fn packed_len(dimension: usize) -> usize {
    dimension * (dimension + 1) / 2
}
