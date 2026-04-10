use covstream::{CovstreamState, MatrixLayout, ShrinkageMode};

fn print_matrix_row_major(label: &str, matrix: &[f64], dimension: usize) {
    println!("{label}");

    for row in 0..dimension {
        let start = row * dimension;
        let end = start + dimension;
        println!("  {:?}", &matrix[start..end]);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimension = 4;
    let mut state = CovstreamState::new(dimension, MatrixLayout::RowMajor)?;

    let samples = [
        [0.0020, -0.0010, 0.0005, 0.0015],
        [0.0010, -0.0005, 0.0007, 0.0010],
        [-0.0008, 0.0004, -0.0003, -0.0006],
        [0.0015, -0.0008, 0.0009, 0.0011],
        [0.0007, -0.0002, 0.0004, 0.0006],
        [-0.0012, 0.0009, -0.0004, -0.0007],
    ];

    println!(
        "Streaming {} samples into a {}-asset state.",
        samples.len(),
        dimension
    );

    for sample in samples {
        state.observe(&sample)?;
    }

    println!("Observed {} samples.", state.sample_count());

    let covariance = state.covariance_buffer()?;
    let shrunk = state.ledoit_wolf_buffer(ShrinkageMode::ClippedAlpha(0.20))?;

    print_matrix_row_major("Sample covariance (row-major):", &covariance, dimension);
    print_matrix_row_major(
        "Ledoit-Wolf style shrinkage toward μI (alpha = 0.20):",
        &shrunk,
        dimension,
    );

    let mut packed_state = CovstreamState::new(dimension, MatrixLayout::UpperTrianglePacked)?;
    for sample in samples {
        packed_state.observe(&sample)?;
    }

    let mut packed_out = vec![0.0; MatrixLayout::UpperTrianglePacked.output_size(dimension)];
    packed_state.ledoit_wolf_buffer_into(ShrinkageMode::ClippedAlpha(0.20), &mut packed_out)?;

    println!("Packed upper-triangle shrinkage buffer:");
    println!("  {:?}", packed_out);

    packed_state.reset();
    println!(
        "After reset, sample count is {}.",
        packed_state.sample_count()
    );

    Ok(())
}
