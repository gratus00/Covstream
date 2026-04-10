#[inline]
pub(crate) fn packed_len(dimension: usize) -> usize {
    dimension * (dimension + 1) / 2
}

#[inline]
pub(crate) fn packed_index(dimension: usize, row: usize, col: usize) -> usize {
    debug_assert!(row <= col);
    debug_assert!(col < dimension);

    row * dimension - (row * row.saturating_sub(1)) / 2 + (col - row)
}
