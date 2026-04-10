mod common;

use covstream::{CovstreamCore, CovstreamState, MatrixLayout, ShrinkageMode};

#[test]
fn ledoit_wolf_row_major_with_zero_alpha_matches_covariance() {
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
    let shrunk = state.ledoit_wolf_row_major(ShrinkageMode::FixedAlpha(0.0));

    match (covariance, shrunk) {
        (Ok(covariance_matrix), Ok(shrunk_matrix)) => {
            assert_eq!(shrunk_matrix, covariance_matrix);
        }
        (Err(err), _) => panic!("expected covariance success, got error: {:?}", err),
        (_, Err(err)) => panic!("expected shrinkage success, got error: {:?}", err),
    }
}

#[test]
fn ledoit_wolf_row_major_with_one_alpha_matches_target() {
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

    let shrunk = state.ledoit_wolf_row_major(ShrinkageMode::FixedAlpha(1.0));

    match shrunk {
        Ok(matrix) => {
            assert_eq!(matrix, vec![2.0, 0.0, 0.0, 2.0]);
        }
        Err(err) => panic!("expected shrinkage success, got error: {:?}", err),
    }
}

#[test]
fn state_ledoit_wolf_row_major_with_zero_alpha_matches_covariance() {
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

    let covariance = state.covariance_row_major();
    let shrunk = state.ledoit_wolf_row_major(ShrinkageMode::FixedAlpha(0.0));

    match (covariance, shrunk) {
        (Ok(covariance_matrix), Ok(shrunk_matrix)) => {
            assert_eq!(shrunk_matrix, covariance_matrix);
        }
        (Err(err), _) => panic!("expected covariance success, got error: {:?}", err),
        (_, Err(err)) => panic!("expected shrinkage success, got error: {:?}", err),
    }
}

#[test]
fn state_ledoit_wolf_buffer_uses_row_major_layout() {
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

    let shrunk = state.ledoit_wolf_buffer(ShrinkageMode::FixedAlpha(1.0));

    match shrunk {
        Ok(matrix) => {
            assert_eq!(matrix, vec![2.0, 0.0, 0.0, 2.0]);
        }
        Err(err) => panic!("expected shrinkage success, got error: {:?}", err),
    }
}

#[test]
fn state_ledoit_wolf_buffer_uses_packed_layout() {
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

    let shrunk = state.ledoit_wolf_buffer(ShrinkageMode::FixedAlpha(1.0));

    match shrunk {
        Ok(matrix) => {
            assert_eq!(matrix, vec![2.0, 0.0, 2.0]);
        }
        Err(err) => panic!("expected shrinkage success, got error: {:?}", err),
    }
}

#[test]
fn end_to_end_streaming_covariance_and_shrinkage_example() {
    let result = CovstreamState::new(2, MatrixLayout::RowMajor);
    let mut state = match result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    for sample in [[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]] {
        match state.observe(&sample) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }

    let covariance = state.covariance_buffer();
    let shrunk = state.ledoit_wolf_buffer(ShrinkageMode::FixedAlpha(0.5));

    match (covariance, shrunk) {
        (Ok(covariance_matrix), Ok(shrunk_matrix)) => {
            assert_eq!(covariance_matrix, vec![1.0, -1.0, -1.0, 1.0]);
            assert_eq!(shrunk_matrix, vec![1.0, -0.5, -0.5, 1.0]);
        }
        (Err(err), _) => panic!("expected covariance success, got error: {:?}", err),
        (_, Err(err)) => panic!("expected shrinkage success, got error: {:?}", err),
    }
}

#[test]
fn ledoit_wolf_row_major_matches_decimal_example_with_tolerance() {
    let result = CovstreamCore::new(4);

    let mut state = match result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    let samples = [
        [0.0020, -0.0010, 0.0005, 0.0015],
        [0.0010, -0.0005, 0.0007, 0.0010],
        [-0.0008, 0.0004, -0.0003, -0.0006],
        [0.0015, -0.0008, 0.0009, 0.0011],
        [0.0007, -0.0002, 0.0004, 0.0006],
        [-0.0012, 0.0009, -0.0004, -0.0007],
    ];

    for sample in samples {
        match state.observe(&sample) {
            Ok(()) => {}
            Err(err) => panic!("expected success, got error: {:?}", err),
        }
    }

    let shrunk = state.ledoit_wolf_row_major(ShrinkageMode::ClippedAlpha(0.20));

    match shrunk {
        Ok(matrix) => {
            let expected = vec![
                1.46275e-6,
                -7.36e-7,
                4.944000000000001e-7,
                9.349333333333332e-7,
                -7.36e-7,
                5.902166666666668e-7,
                -2.8320000000000005e-7,
                -5.264000000000001e-7,
                4.944000000000001e-7,
                -2.8320000000000005e-7,
                3.918166666666667e-7,
                3.6319999999999997e-7,
                9.349333333333332e-7,
                -5.264000000000001e-7,
                3.6319999999999997e-7,
                8.475499999999999e-7,
            ];

            common::assert_slice_close(&matrix, &expected, 1e-15);
        }
        Err(err) => panic!("expected shrinkage success, got error: {:?}", err),
    }
}
