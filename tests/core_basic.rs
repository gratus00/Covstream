use covstream::{CovstreamCore, CovstreamError, CovstreamState, MatrixLayout};

#[test]
fn core_new_rejects_zero_dimension() {
    let result = CovstreamCore::new(0);

    match result {
        Err(CovstreamError::ZeroDimension) => {}
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(_) => panic!("expected ZeroDimension error, got success"),
    }
}

#[test]
fn core_new_initializes_zero_state() {
    let result = CovstreamCore::new(3);

    match result {
        Ok(state) => {
            assert_eq!(state.dimension(), 3);
            assert_eq!(state.sample_count(), 0);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn state_new_uses_requested_layout() {
    let result = CovstreamState::new(3, MatrixLayout::RowMajor);

    match result {
        Ok(state) => {
            assert_eq!(state.layout(), MatrixLayout::RowMajor);
            assert_eq!(state.dimension(), 3);
            assert_eq!(state.sample_count(), 0);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn observe_rejects_wrong_dimension() {
    let state_result = CovstreamCore::new(2);

    let mut state = match state_result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    let result = state.observe(&[1.0]);

    match result {
        Err(CovstreamError::WrongDimension { expected, got }) => {
            assert_eq!(expected, 2);
            assert_eq!(got, 1);
        }
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected WrongDimension error, got success"),
    }
}

#[test]
fn observe_rejects_non_finite_input() {
    let state_result = CovstreamCore::new(2);

    let mut state = match state_result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    let result = state.observe(&[1.0, f64::NAN]);

    match result {
        Err(CovstreamError::NonFiniteInput) => {}
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected NonFiniteInput error, got success"),
    }
}

#[test]
fn observe_increments_sample_count() {
    let state_result = CovstreamCore::new(2);

    let mut state = match state_result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    let result = state.observe(&[1.0, 2.0]);

    match result {
        Ok(()) => {
            assert_eq!(state.sample_count(), 1);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn observe_updates_mean() {
    let result = CovstreamCore::new(2);
    let mut state = match result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    let first = state.observe(&[2.0, 4.0]);
    match first {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    assert_eq!(state.mean(), &[2.0, 4.0]);

    let second = state.observe(&[4.0, 8.0]);
    match second {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    assert_eq!(state.mean(), &[3.0, 6.0]);
}

#[test]
fn observe_updates_covariance_numerator() {
    let result = CovstreamCore::new(2);
    let mut state = match result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    match state.observe(&[1.0, 2.0]) {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    assert_eq!(state.cov_numerator(), &[0.0, 0.0, 0.0]);

    match state.observe(&[3.0, 4.0]) {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    assert_eq!(state.cov_numerator(), &[2.0, 2.0, 2.0]);
}
