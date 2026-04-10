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
