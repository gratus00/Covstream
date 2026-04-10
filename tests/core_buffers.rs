use covstream::{CovstreamCore, CovstreamError, ShrinkageMode};

#[test]
fn covariance_row_major_into_writes_expected_values() {
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

    let mut out = vec![0.0; 4];
    let result = state.covariance_row_major_into(&mut out);

    match result {
        Ok(()) => {
            assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn covariance_row_major_into_rejects_small_output_buffer() {
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

    let mut out = vec![0.0; 3];
    let result = state.covariance_row_major_into(&mut out);

    match result {
        Err(CovstreamError::OutputBufferTooSmall { expected, got }) => {
            assert_eq!(expected, 4);
            assert_eq!(got, 3);
        }
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected OutputBufferTooSmall error, got success"),
    }
}

#[test]
fn covariance_upper_triangle_packed_into_writes_expected_values() {
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

    let mut out = vec![0.0; 3];
    let result = state.covariance_upper_triangle_packed_into(&mut out);

    match result {
        Ok(()) => {
            assert_eq!(out, vec![2.0, 2.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn covariance_upper_triangle_packed_into_rejects_small_output_buffer() {
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

    let mut out = vec![0.0; 2];
    let result = state.covariance_upper_triangle_packed_into(&mut out);

    match result {
        Err(CovstreamError::OutputBufferTooSmall { expected, got }) => {
            assert_eq!(expected, 3);
            assert_eq!(got, 2);
        }
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected OutputBufferTooSmall error, got success"),
    }
}

#[test]
fn ledoit_wolf_row_major_into_writes_expected_values() {
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

    let mut out = vec![0.0; 4];
    let result = state.ledoit_wolf_row_major_into(ShrinkageMode::FixedAlpha(1.0), &mut out);

    match result {
        Ok(()) => {
            assert_eq!(out, vec![2.0, 0.0, 0.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn ledoit_wolf_row_major_into_rejects_small_output_buffer() {
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

    let mut out = vec![0.0; 3];
    let result = state.ledoit_wolf_row_major_into(ShrinkageMode::FixedAlpha(1.0), &mut out);

    match result {
        Err(CovstreamError::OutputBufferTooSmall { expected, got }) => {
            assert_eq!(expected, 4);
            assert_eq!(got, 3);
        }
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected OutputBufferTooSmall error, got success"),
    }
}

#[test]
fn ledoit_wolf_upper_triangle_packed_into_writes_expected_values() {
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

    let mut out = vec![0.0; 3];
    let result =
        state.ledoit_wolf_upper_triangle_packed_into(ShrinkageMode::FixedAlpha(1.0), &mut out);

    match result {
        Ok(()) => {
            assert_eq!(out, vec![2.0, 0.0, 2.0]);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn ledoit_wolf_upper_triangle_packed_into_rejects_small_output_buffer() {
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

    let mut out = vec![0.0; 2];
    let result =
        state.ledoit_wolf_upper_triangle_packed_into(ShrinkageMode::FixedAlpha(1.0), &mut out);

    match result {
        Err(CovstreamError::OutputBufferTooSmall { expected, got }) => {
            assert_eq!(expected, 3);
            assert_eq!(got, 2);
        }
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected OutputBufferTooSmall error, got success"),
    }
}
