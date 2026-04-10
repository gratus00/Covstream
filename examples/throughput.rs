use std::env;
use std::time::Instant;

use covstream::{CovstreamState, MatrixLayout, ShrinkageMode};

fn parse_arg(args: &[String], index: usize, default: usize) -> usize {
    args.get(index)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn next_unit(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let bits = *seed >> 11;
    let unit = (bits as f64) / ((1_u64 << 53) as f64);
    0.004 * (unit - 0.5)
}

fn fill_sample(sample: &mut [f64], seed: &mut u64) {
    for value in sample {
        *value = next_unit(seed);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let dimension = parse_arg(&args, 1, 64);
    let sample_count = parse_arg(&args, 2, 100_000);

    let mut state = CovstreamState::new(dimension, MatrixLayout::UpperTrianglePacked)?;
    let mut sample = vec![0.0; dimension];
    let mut covariance = vec![0.0; MatrixLayout::UpperTrianglePacked.output_size(dimension)];
    let mut shrunk = vec![0.0; MatrixLayout::UpperTrianglePacked.output_size(dimension)];
    let mut seed = 0x5eed_u64;

    println!("Streaming {sample_count} synthetic samples for dimension {dimension}.");
    println!("Run with: cargo run --release --example throughput -- <dimension> <samples>");

    let observe_start = Instant::now();
    for _ in 0..sample_count {
        fill_sample(&mut sample, &mut seed);
        state.observe(&sample)?;
    }
    let observe_elapsed = observe_start.elapsed();

    let covariance_start = Instant::now();
    state.covariance_buffer_into(&mut covariance)?;
    let covariance_elapsed = covariance_start.elapsed();

    let shrinkage_start = Instant::now();
    state.ledoit_wolf_buffer_into(ShrinkageMode::ClippedAlpha(0.20), &mut shrunk)?;
    let shrinkage_elapsed = shrinkage_start.elapsed();

    let samples_per_second = sample_count as f64 / observe_elapsed.as_secs_f64();

    println!("Observed {} samples.", state.sample_count());
    println!(
        "Observe time: {:?} ({:.2} samples/sec)",
        observe_elapsed, samples_per_second
    );
    println!("Covariance extraction time: {:?}", covariance_elapsed);
    println!("Shrinkage extraction time: {:?}", shrinkage_elapsed);
    println!(
        "Packed output length: {} entries",
        MatrixLayout::UpperTrianglePacked.output_size(dimension)
    );
    println!(
        "First packed covariance entries: {:?}",
        &covariance[..covariance.len().min(8)]
    );
    println!(
        "First packed shrunk entries: {:?}",
        &shrunk[..shrunk.len().min(8)]
    );

    Ok(())
}
