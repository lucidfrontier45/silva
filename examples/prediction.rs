use std::{
    fs::{File, read_to_string},
    io::{BufWriter, Write},
    path::Path,
};

use silva::MultiOutputForest;

pub fn read_features(path: impl AsRef<Path>) -> Vec<Vec<f64>> {
    let x_content = read_to_string(path).expect("failed to read file");
    x_content
        .lines()
        .map(|line| {
            line.split(',')
                .map(|s| s.parse::<f64>().expect("Failed to parse X value"))
                .collect()
        })
        .collect()
}

fn main() {
    // get model file and X.csv from command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "Usage: {} <model_file> <X.csv> <output_predictions.csv>",
            args[0]
        );
        std::process::exit(1);
    }

    let model_file = &args[1];
    let x_csv = &args[2];
    let output_csv = &args[3];

    // load model
    let model = MultiOutputForest::from_file(model_file).expect("Failed to load model");

    // load features
    let x_data = read_features(x_csv);

    let mut writer =
        BufWriter::new(File::create(output_csv).expect("Failed to create output file"));

    for x in &x_data {
        let pred = model.predict(x);
        let line = pred
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(writer, "{}", line).expect("Failed to write prediction");
    }
}
