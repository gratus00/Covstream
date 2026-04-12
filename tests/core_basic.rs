mod common;

use common::assert_slice_close;
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
fn observe_batch_row_major_increments_sample_count() {
    let state_result = CovstreamCore::new(2);

    let mut state = match state_result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    let result = state.observe_batch_row_major(&[1.0, 2.0, 3.0, 4.0]);

    match result {
        Ok(()) => {
            assert_eq!(state.sample_count(), 2);
        }
        Err(err) => panic!("expected success, got error: {:?}", err),
    }
}

#[test]
fn observe_batch_row_major_rejects_malformed_input() {
    let state_result = CovstreamCore::new(2);

    let mut state = match state_result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    let result = state.observe_batch_row_major(&[1.0, 2.0, 3.0]);

    match result {
        Err(CovstreamError::MalformedBatchInput { dimension, len }) => {
            assert_eq!(dimension, 2);
            assert_eq!(len, 3);
        }
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected MalformedBatchInput error, got success"),
    }
}

#[test]
fn observe_batch_row_major_rejects_non_finite_input() {
    let state_result = CovstreamCore::new(2);

    let mut state = match state_result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    let result = state.observe_batch_row_major(&[1.0, 2.0, 3.0, f64::NAN]);

    match result {
        Err(CovstreamError::NonFiniteInput) => {}
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected NonFiniteInput error, got success"),
    }
}

#[test]
fn observe_batch_row_major_matches_repeated_observe() {
    let batch_result = CovstreamCore::new(2);
    let mut batch_state = match batch_result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    match batch_state.observe_batch_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    let single_result = CovstreamCore::new(2);
    let mut single_state = match single_result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    match single_state.observe(&[1.0, 2.0]) {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    match single_state.observe(&[3.0, 4.0]) {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    match single_state.observe(&[5.0, 6.0]) {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    assert_eq!(batch_state.sample_count(), single_state.sample_count());
    assert_eq!(batch_state.mean(), single_state.mean());
    assert_eq!(batch_state.cov_numerator(), single_state.cov_numerator());
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

#[test]
fn core_reset_clears_state_without_reallocating() {
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

    state.reset();

    assert_eq!(state.sample_count(), 0);
    assert_eq!(state.mean(), &[0.0, 0.0]);
    assert_eq!(state.cov_numerator(), &[0.0, 0.0, 0.0]);
}

#[test]
fn state_reset_clears_wrapper_state() {
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

    state.reset();

    assert_eq!(state.sample_count(), 0);
}

#[test]
fn state_observe_batch_row_major_works() {
    let result = CovstreamState::new(2, MatrixLayout::RowMajor);
    let mut state = match result {
        Ok(state) => state,
        Err(err) => panic!("expected success, got error: {:?}", err),
    };

    match state.observe_batch_row_major(&[1.0, 2.0, 3.0, 4.0]) {
        Ok(()) => {}
        Err(err) => panic!("expected success, got error: {:?}", err),
    }

    assert_eq!(state.sample_count(), 2);
}

#[test]
fn merge_matches_repeated_observe_with_tolerance() {
    let samples = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let mut left = CovstreamCore::new(2).expect("valid dimension");
    left.observe(&samples[0]).expect("finite sample");
    left.observe(&samples[1]).expect("finite sample");

    let mut right = CovstreamCore::new(2).expect("valid dimension");
    right.observe(&samples[2]).expect("finite sample");
    right.observe(&samples[3]).expect("finite sample");

    let mut merged = left.clone();
    merged.merge(&right).expect("matching dimensions");

    let mut serial = CovstreamCore::new(2).expect("valid dimension");
    for sample in samples {
        serial.observe(&sample).expect("finite sample");
    }

    assert_eq!(merged.sample_count(), serial.sample_count());
    assert_slice_close(merged.mean(), serial.mean(), 1e-12);
    assert_slice_close(merged.cov_numerator(), serial.cov_numerator(), 1e-10);
}

#[test]
fn parallel_batch_matches_serial_batch_with_tolerance() {
    let batch = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let mut serial = CovstreamCore::new(2).expect("valid dimension");
    serial
        .observe_batch_row_major(&batch)
        .expect("finite batch");

    let mut parallel = CovstreamCore::new(2).expect("valid dimension");
    parallel
        .observe_batch_row_major_parallel(&batch)
        .expect("finite batch");

    assert_eq!(parallel.sample_count(), serial.sample_count());
    assert_slice_close(parallel.mean(), serial.mean(), 1e-12);
    assert_slice_close(parallel.cov_numerator(), serial.cov_numerator(), 1e-10);
}

#[test]
fn merge_rejects_wrong_dimension() {
    let mut left = CovstreamCore::new(2).expect("valid dimension");
    let right = CovstreamCore::new(3).expect("valid dimension");

    let result = left.merge(&right);

    match result {
        Err(CovstreamError::WrongDimension { expected, got }) => {
            assert_eq!(expected, 2);
            assert_eq!(got, 3);
        }
        Err(other) => panic!("unexpected error: {:?}", other),
        Ok(()) => panic!("expected WrongDimension error, got success"),
    }
}

#[test]
fn merge_into_empty_state_copies_other_state() {
    let mut source = CovstreamCore::new(2).expect("valid dimension");
    source.observe(&[1.0, 2.0]).expect("finite sample");
    source.observe(&[3.0, 6.0]).expect("finite sample");
    source.observe(&[5.0, 10.0]).expect("finite sample");

    let mut target = CovstreamCore::new(2).expect("valid dimension");
    target.merge(&source).expect("matching dimensions");

    assert_eq!(target.sample_count(), source.sample_count());
    assert_slice_close(target.mean(), source.mean(), 1e-12);
    assert_slice_close(target.cov_numerator(), source.cov_numerator(), 1e-12);
}

#[test]
fn parallel_trusted_batch_matches_serial_trusted_batch_with_tolerance() {
    let batch: Vec<f64> = (0..(64 * 32))
        .map(|i| ((i % 32) as f64 + 1.0) * ((i / 32) as f64 + 1.0) * 0.001)
        .collect();

    let mut serial = CovstreamCore::new(32).expect("valid dimension");
    serial
        .observe_batch_row_major_trusted_finite(&batch)
        .expect("well-shaped batch");

    let mut parallel = CovstreamCore::new(32).expect("valid dimension");
    parallel
        .observe_batch_row_major_parallel_trusted_finite(&batch)
        .expect("well-shaped batch");

    assert_eq!(parallel.sample_count(), serial.sample_count());
    assert_slice_close(parallel.mean(), serial.mean(), 1e-12);
    assert_slice_close(parallel.cov_numerator(), serial.cov_numerator(), 1e-10);
}

#[test]
fn state_parallel_batch_matches_serial_batch_buffer() {
    let batch: Vec<f64> = (0..(32 * 16))
        .map(|i| ((i % 16) as f64 + 1.0) * ((i / 16) as f64 + 1.0) * 0.01)
        .collect();

    let mut serial = CovstreamState::new(16, MatrixLayout::RowMajor).expect("valid state");
    serial
        .observe_batch_row_major(&batch)
        .expect("finite batch");

    let mut parallel = CovstreamState::new(16, MatrixLayout::RowMajor).expect("valid state");
    parallel
        .observe_batch_row_major_parallel(&batch)
        .expect("finite batch");

    let serial_cov = serial
        .covariance_buffer()
        .expect("enough samples for covariance");
    let parallel_cov = parallel
        .covariance_buffer()
        .expect("enough samples for covariance");

    assert_eq!(parallel.sample_count(), serial.sample_count());
    assert_slice_close(&parallel_cov, &serial_cov, 1e-10);
}
