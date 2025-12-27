use std::path::Path;

use anyhow::Result as AnyResult;

use crate::{MultiOutputForest, tree::Tree};

pub fn read_lightgbm_model(path: impl AsRef<Path>) -> AnyResult<MultiOutputForest> {
    todo!()
}

#[derive(Clone)]
struct LGBMTreeRecord {
    split_features: Vec<usize>,
    thresholds: Vec<f64>,
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    left_values: Vec<f64>,
}

impl From<LGBMTreeRecord> for Tree {
    fn from(record: LGBMTreeRecord) -> Self {
        todo!()
    }
}

fn read_lightgbm_txt(path: impl AsRef<Path>) -> AnyResult<Vec<Vec<LGBMTreeRecord>>> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();

    let mut num_tree_per_iteration: Option<usize> = None;
    let mut tree_records: Vec<LGBMTreeRecord> = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if line.starts_with("Tree=") {
            if let Some(record) = parse_tree_section(&lines, line_idx) {
                tree_records.push(record);
            }
        } else if let Some((key, value)) = line.split_once('=') {
            match key {
                "num_tree_per_iteration" => {
                    num_tree_per_iteration = Some(value.parse()?);
                }
                _ => {}
            }
        }
    }

    let num_trees = num_tree_per_iteration
        .ok_or_else(|| anyhow::anyhow!("num_tree_per_iteration not found in header"))?;

    let num_iterations = tree_records.len() / num_trees;
    let mut result: Vec<Vec<LGBMTreeRecord>> = vec![Vec::new(); num_trees];

    for i in 0..num_iterations {
        for j in 0..num_trees {
            result[j].push(tree_records[i * num_trees + j].clone());
        }
    }

    Ok(result)
}

fn parse_tree_section(lines: &[&str], start_idx: usize) -> Option<LGBMTreeRecord> {
    let mut split_features: Option<Vec<usize>> = None;
    let mut thresholds: Option<Vec<f64>> = None;
    let mut left_children: Option<Vec<i32>> = None;
    let mut right_children: Option<Vec<i32>> = None;
    let mut leaf_values: Option<Vec<f64>> = None;

    let mut idx = start_idx + 1;

    while idx < lines.len() {
        let line = lines[idx].trim();

        if line.is_empty() || line.starts_with("Tree=") {
            break;
        }

        if let Some((key, value)) = line.split_once('=') {
            match key {
                "split_feature" => {
                    split_features = Some(
                        value
                            .split_whitespace()
                            .map(|s| s.parse().ok())
                            .collect::<Option<Vec<_>>>()?,
                    );
                }
                "threshold" => {
                    thresholds = Some(
                        value
                            .split_whitespace()
                            .map(|s| s.parse().ok())
                            .collect::<Option<Vec<_>>>()?,
                    );
                }
                "left_child" => {
                    left_children = Some(
                        value
                            .split_whitespace()
                            .map(|s| s.parse().ok())
                            .collect::<Option<Vec<_>>>()?,
                    );
                }
                "right_child" => {
                    right_children = Some(
                        value
                            .split_whitespace()
                            .map(|s| s.parse().ok())
                            .collect::<Option<Vec<_>>>()?,
                    );
                }
                "leaf_value" => {
                    leaf_values = Some(
                        value
                            .split_whitespace()
                            .map(|s| s.parse().ok())
                            .collect::<Option<Vec<_>>>()?,
                    );
                }
                _ => {}
            }
        }

        idx += 1;
    }

    Some(LGBMTreeRecord {
        split_features: split_features?,
        thresholds: thresholds?,
        left_children: left_children?,
        right_children: right_children?,
        left_values: leaf_values?,
    })
}
