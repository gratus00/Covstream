use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use covstream::{CovstreamCore, MatrixLayout, ShrinkageMode};

fn sample_for_index(dimension: usize, index: usize) -> Vec<f64> {
    let scale = (index + 1) as f64;
    (0..dimension)
        .map(|i| ((i + 1) as f64) * scale * 0.001)
        .collect()
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

    for &dimension in &[2usize, 8, 32, 64, 128, 256] {
        let sample = sample_for_index(dimension, 0);
        group.bench_with_input(BenchmarkId::from_parameter(dimension), &dimension, |b, &k| {
            b.iter(|| {
                let mut state = CovstreamCore::new(k).expect("valid dimension");
                state.observe(black_box(&sample)).expect("finite sample");
                black_box(state);
            });
        });
    }

    group.finish();
}

fn bench_covariance_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("covariance_extract");

    for &dimension in &[2usize, 8, 32, 64, 128, 256] {
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

    for &dimension in &[2usize, 8, 32, 64, 128, 256] {
        let state = seeded_state(dimension, 64);
        let mut row_major = vec![0.0; MatrixLayout::RowMajor.output_size(dimension)];

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
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_observe,
    bench_covariance_extract,
    bench_shrinkage_extract
);
criterion_main!(benches);
