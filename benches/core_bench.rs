use covstream::{CovstreamCore, MatrixLayout, ShrinkageMode};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::time::{Duration, Instant};

fn sample_for_index(dimension: usize, index: usize) -> Vec<f64> {
    let scale = (index + 1) as f64;
    (0..dimension)
        .map(|i| ((i + 1) as f64) * scale * 0.001)
        .collect()
}

fn batch_for_count(dimension: usize, samples: usize) -> Vec<f64> {
    let mut batch = Vec::with_capacity(dimension * samples);

    for index in 0..samples {
        batch.extend(sample_for_index(dimension, index));
    }

    batch
}

fn seeded_state(dimension: usize, samples: usize) -> CovstreamCore {
    let mut state = CovstreamCore::new(dimension).expect("valid dimension");

    for index in 0..samples {
        let sample = sample_for_index(dimension, index);
        state.observe(&sample).expect("finite sample");
    }

    state
}

fn bench_observe(c: &mut Criterion) {
    let mut group = c.benchmark_group("observe");

    for &dimension in &[2usize, 8, 32, 64, 128, 256, 512] {
        let sample = sample_for_index(dimension, 0);
        group.bench_with_input(
            BenchmarkId::from_parameter(dimension),
            &dimension,
            |b, &k| {
                b.iter(|| {
                    let mut state = CovstreamCore::new(k).expect("valid dimension");
                    state.observe(black_box(&sample)).expect("finite sample");
                    black_box(state);
                });
            },
        );
    }

    group.finish();
}

fn bench_observe_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("observe_batch");

    for &dimension in &[8usize, 32, 128, 256] {
        for &sample_count in &[8usize, 64, 256] {
            let batch = batch_for_count(dimension, sample_count);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("single_calls/d{}_n{}", dimension, sample_count),
                    dimension,
                ),
                &(dimension, sample_count),
                |b, &(k, n)| {
                    b.iter_batched(
                        || CovstreamCore::new(k).expect("valid dimension"),
                        |mut state| {
                            for sample in batch.chunks_exact(k).take(n) {
                                state.observe(black_box(sample)).expect("finite sample");
                            }
                            black_box(state);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("batch_call/d{}_n{}", dimension, sample_count),
                    dimension,
                ),
                &(dimension, sample_count),
                |b, &(k, _)| {
                    b.iter_batched(
                        || CovstreamCore::new(k).expect("valid dimension"),
                        |mut state| {
                            state
                                .observe_batch_row_major(black_box(&batch))
                                .expect("finite batch");
                            black_box(state);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

fn bench_observe_batch_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("observe_batch_parallel");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));

    for &dimension in &[32usize, 128, 256, 512] {
        for &sample_count in &[256usize, 1024] {
            let batch = batch_for_count(dimension, sample_count);
            let input = (dimension, sample_count);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("checked_serial/d{}_n{}", dimension, sample_count),
                    dimension,
                ),
                &input,
                |b, &(k, _)| {
                    b.iter_batched(
                        || CovstreamCore::new(k).expect("valid dimension"),
                        |mut state| {
                            state
                                .observe_batch_row_major(black_box(&batch))
                                .expect("finite batch");
                            black_box(state);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("trusted_serial/d{}_n{}", dimension, sample_count),
                    dimension,
                ),
                &input,
                |b, &(k, _)| {
                    b.iter_batched(
                        || CovstreamCore::new(k).expect("valid dimension"),
                        |mut state| {
                            state
                                .observe_batch_row_major_trusted_finite(black_box(&batch))
                                .expect("well-shaped batch");
                            black_box(state);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("checked_parallel/d{}_n{}", dimension, sample_count),
                    dimension,
                ),
                &input,
                |b, &(k, _)| {
                    b.iter_batched(
                        || CovstreamCore::new(k).expect("valid dimension"),
                        |mut state| {
                            state
                                .observe_batch_row_major_parallel(black_box(&batch))
                                .expect("finite batch");
                            black_box(state);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("trusted_parallel/d{}_n{}", dimension, sample_count),
                    dimension,
                ),
                &input,
                |b, &(k, _)| {
                    b.iter_batched(
                        || CovstreamCore::new(k).expect("valid dimension"),
                        |mut state| {
                            state
                                .observe_batch_row_major_parallel_trusted_finite(black_box(&batch))
                                .expect("well-shaped batch");
                            black_box(state);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

fn bench_covariance_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("covariance_extract");

    for &dimension in &[2usize, 8, 32, 64, 128, 256, 512] {
        let state = seeded_state(dimension, 64);
        let mut row_major = vec![0.0; MatrixLayout::RowMajor.output_size(dimension)];
        let mut packed = vec![0.0; MatrixLayout::UpperTrianglePacked.output_size(dimension)];

        group.bench_with_input(
            BenchmarkId::new("row_major_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    state
                        .covariance_row_major_into(black_box(&mut row_major))
                        .expect("enough samples");
                    black_box(&row_major);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("packed_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    state
                        .covariance_upper_triangle_packed_into(black_box(&mut packed))
                        .expect("enough samples");
                    black_box(&packed);
                });
            },
        );
    }

    group.finish();
}

fn bench_shrinkage_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinkage_extract");

    for &dimension in &[2usize, 8, 32, 64, 128, 256, 512] {
        let state = seeded_state(dimension, 64);
        let mut row_major = vec![0.0; MatrixLayout::RowMajor.output_size(dimension)];
        let mut packed = vec![0.0; MatrixLayout::UpperTrianglePacked.output_size(dimension)];

        group.bench_with_input(
            BenchmarkId::new("ledoit_wolf_row_major_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    state
                        .ledoit_wolf_row_major_into(
                            ShrinkageMode::ClippedAlpha(0.2),
                            black_box(&mut row_major),
                        )
                        .expect("enough samples");
                    black_box(&row_major);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ledoit_wolf_packed_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    state
                        .ledoit_wolf_upper_triangle_packed_into(
                            ShrinkageMode::ClippedAlpha(0.2),
                            black_box(&mut packed),
                        )
                        .expect("enough samples");
                    black_box(&packed);
                });
            },
        );
    }

    group.finish();
}

fn bench_api_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_shape");

    for &dimension in &[32usize, 128, 256] {
        let state = seeded_state(dimension, 64);
        let mut row_major = vec![0.0; MatrixLayout::RowMajor.output_size(dimension)];

        group.bench_with_input(
            BenchmarkId::new("covariance_vec", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let matrix = state.covariance_row_major().expect("enough samples");
                    black_box(matrix);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("covariance_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    state
                        .covariance_row_major_into(black_box(&mut row_major))
                        .expect("enough samples");
                    black_box(&row_major);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("shrinkage_vec", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let matrix = state
                        .ledoit_wolf_row_major(ShrinkageMode::ClippedAlpha(0.2))
                        .expect("enough samples");
                    black_box(matrix);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("shrinkage_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    state
                        .ledoit_wolf_row_major_into(
                            ShrinkageMode::ClippedAlpha(0.2),
                            black_box(&mut row_major),
                        )
                        .expect("enough samples");
                    black_box(&row_major);
                });
            },
        );
    }

    group.finish();
}

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");

    for &dimension in &[32usize, 128, 256] {
        group.bench_with_input(
            BenchmarkId::from_parameter(dimension),
            &dimension,
            |b, &k| {
                b.iter_batched(
                    || {
                        let state = CovstreamCore::new(k).expect("valid dimension");
                        let out = vec![0.0; MatrixLayout::UpperTrianglePacked.output_size(k)];
                        (state, out)
                    },
                    |(mut state, mut out)| {
                        for index in 0..64 {
                            let sample = sample_for_index(k, index);
                            state.observe(black_box(&sample)).expect("finite sample");

                            if index >= 1 && index % 16 == 15 {
                                state
                                    .ledoit_wolf_upper_triangle_packed_into(
                                        ShrinkageMode::ClippedAlpha(0.2),
                                        black_box(&mut out),
                                    )
                                    .expect("enough samples");
                            }
                        }

                        black_box((state, out));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_observe_hot(c: &mut Criterion) {
    let mut group = c.benchmark_group("observe_hot");

    for &dimension in &[8usize, 32, 128, 256, 512] {
        let batch = batch_for_count(dimension, 2048);

        group.bench_with_input(
            BenchmarkId::new("trusted", dimension),
            &dimension,
            |b, &k| {
                b.iter_custom(|iters| {
                    let mut state = seeded_state(k, 64);
                    let mut sample_index = 0usize;
                    let sample_count = batch.len() / k;
                    let start = Instant::now();

                    for _ in 0..iters {
                        let offset = sample_index * k;
                        state
                            .observe_trusted_finite(black_box(&batch[offset..offset + k]))
                            .expect("finite sample");
                        sample_index = (sample_index + 1) % sample_count;
                    }

                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_observe,
    bench_observe_batch,
    bench_observe_batch_parallel,
    bench_covariance_extract,
    bench_shrinkage_extract,
    bench_api_shape,
    bench_mixed_workload,
    bench_observe_hot,
);
criterion_main!(benches);
