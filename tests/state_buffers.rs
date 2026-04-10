use covstream::{CovstreamState, MatrixLayout, ShrinkageMode};

#[test]
fn state_covariance_buffer_into_uses_row_major_layout() {
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

    let mut out = vec![0.0; 4];
    let result = state.covariance_buffer_into(&mut out);

    match result {
        Ok(()) => {
            assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn state_covariance_buffer_into_uses_packed_layout() {
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

    let mut out = vec![0.0; 3];
    let result = state.covariance_buffer_into(&mut out);

    match result {
        Ok(()) => {
            assert_eq!(out, vec![2.0, 2.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn state_ledoit_wolf_buffer_into_uses_row_major_layout() {
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

    let mut out = vec![0.0; 4];
    let result = state.ledoit_wolf_buffer_into(ShrinkageMode::FixedAlpha(1.0), &mut out);

    match result {
        Ok(()) => {
            assert_eq!(out, vec![2.0, 0.0, 0.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn state_ledoit_wolf_buffer_into_uses_packed_layout() {
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

    let mut out = vec![0.0; 3];
    let result = state.ledoit_wolf_buffer_into(ShrinkageMode::FixedAlpha(1.0), &mut out);

    match result {
        Ok(()) => {
            assert_eq!(out, vec![2.0, 0.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}
