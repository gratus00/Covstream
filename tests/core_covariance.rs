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
