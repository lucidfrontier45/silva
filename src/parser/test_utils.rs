use std::{fs::read_to_string, path::Path};

use crate::MultiOutputForest;
use anyhow::Result as AnyResult;

pub fn all_close(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        if (x - y).abs() > tol {
            return false;
        }
    }
    true
}

pub fn read_features(path: &Path) -> Vec<Vec<f64>> {
    let Ok(x_content) = read_to_string(path) else {
        panic!("Failed to read X from{:?}", path);
    };
    x_content
        .lines()
        .map(|line| {
            line.split(',')
                .map(|s| s.parse::<f64>().expect("Failed to parse X value"))
                .collect()
        })
        .collect()
}

pub fn read_labels_flattened(path: &Path) -> Vec<f64> {
    let Ok(y_content) = read_to_string(path) else {
        panic!("Failed to read y from {:?}", path);
    };
    y_content
        .lines()
        .flat_map(|line| {
            line.split(',')
                .map(|s| s.parse::<f64>().expect("Failed to parse y value"))
                .collect::<Vec<f64>>()
        })
        .collect()
}

pub fn test_model_prediction(
    data_dir: &Path,
    forest: &MultiOutputForest,
    tolerance: f64,
) -> AnyResult<()> {
    let x_path = data_dir.join("X.csv");
    let x_data = read_features(&x_path);

    let y_path = data_dir.join("y.csv");
    let y_true = read_labels_flattened(&y_path);

    let y_pred = x_data
        .iter()
        .flat_map(|x| forest.predict(x))
        .map(|v| v.into_inner())
        .collect::<Vec<f64>>();

    assert!(
        all_close(&y_pred, &y_true, tolerance),
        "Predictions and y values differ more than tolerance"
    );

    Ok(())
}
