mod common;

use covstream::{CovstreamCore, CovstreamError, CovstreamState, MatrixLayout};

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

    let covariance = state.covariance_buffer();

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

    let covariance = state.covariance_buffer();

    match covariance {
        Ok(matrix) => {
            assert_eq!(matrix, vec![2.0, 2.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn covariance_row_major_matches_decimal_example_with_tolerance() {
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

    let covariance = state.covariance_row_major();

    match covariance {
        Ok(matrix) => {
            let expected = vec![
                1.6226666666666665e-6,
                -9.2e-7,
                6.180000000000001e-7,
                1.1686666666666665e-6,
                -9.2e-7,
                5.32e-7,
                -3.54e-7,
                -6.580000000000001e-7,
                6.180000000000001e-7,
                -3.54e-7,
                2.84e-7,
                4.5399999999999996e-7,
                1.1686666666666665e-6,
                -6.580000000000001e-7,
                4.5399999999999996e-7,
                8.536666666666666e-7,
            ];

            common::assert_slice_close(&matrix, &expected, 1e-15);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}
