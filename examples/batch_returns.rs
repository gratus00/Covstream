use covstream::{CovstreamState, MatrixLayout, ShrinkageMode};

const RETURNS_CSV: &str = "\
asset_a,asset_b,asset_c,asset_d
0.0020,-0.0010,0.0005,0.0015
0.0010,-0.0005,0.0007,0.0010
-0.0008,0.0004,-0.0003,-0.0006
0.0015,-0.0008,0.0009,0.0011
0.0007,-0.0002,0.0004,0.0006
-0.0012,0.0009,-0.0004,-0.0007
";

struct ParsedReturns {
    assets: Vec<String>,
    flat_samples: Vec<f64>,
    sample_count: usize,
}

fn parse_returns_csv(csv: &str) -> Result<ParsedReturns, Box<dyn std::error::Error>> {
    let mut lines = csv.lines().filter(|line| !line.trim().is_empty());

    let header_line = lines.next().ok_or("missing header line")?;
    let headers: Vec<String> = header_line
        .split(',')
        .map(|entry| entry.trim().to_string())
        .collect();

    if headers.is_empty() {
        return Err("header must contain at least one column".into());
    }

    let dimension = headers.len();
    let mut flat_samples = Vec::new();
    let mut sample_count = 0;

    for line in lines {
        let row: Vec<f64> = line
            .split(',')
            .map(|entry| entry.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;

        if row.len() != dimension {
            return Err(format!(
                "row {} has length {}, expected {}",
                sample_count + 1,
                row.len(),
                dimension
            )
            .into());
        }

        flat_samples.extend(row);
        sample_count += 1;
    }

    Ok(ParsedReturns {
        assets: headers,
        flat_samples,
        sample_count,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parsed = parse_returns_csv(RETURNS_CSV)?;
    let assets = parsed.assets;
    let flat_returns = parsed.flat_samples;
    let sample_count = parsed.sample_count;
    let dimension = assets.len();

    let mut state = CovstreamState::new(dimension, MatrixLayout::UpperTrianglePacked)?;
    state.observe_batch_row_major(&flat_returns)?;

    let covariance = state.covariance_buffer()?;
    let shrunk = state.ledoit_wolf_buffer(ShrinkageMode::ClippedAlpha(0.20))?;

    println!(
        "Loaded {} aligned return vectors for {} assets.",
        sample_count, dimension
    );
    println!("Assets: {:?}", assets);
    println!("Packed covariance length: {}", covariance.len());
    println!("Packed shrunk covariance length: {}", shrunk.len());
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
