use std::path::Path;

use anyhow::Result as AnyResult;

use crate::{
    Forest, MultiOutputForest,
    tree::{Tree, TreeNode},
};

pub fn read_lightgbm_model(path: impl AsRef<Path>) -> AnyResult<MultiOutputForest> {
    let tree_records = read_lightgbm_txt(path)?;
    let trees = tree_records
        .into_iter()
        .map(|records| records.into_iter().map(Tree::from).collect::<Vec<Tree>>())
        .collect::<Vec<Vec<Tree>>>();
    let forests = trees
        .into_iter()
        .map(|tree_vec| Forest::new(0.0, tree_vec))
        .collect::<Vec<Forest>>();
    Ok(MultiOutputForest::new(forests))
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
        use ordered_float::NotNan;

        let mut nodes = Vec::new();

        for i in 0..record.split_features.len() {
            let left_child = record.left_children[i];
            let right_child = record.right_children[i];

            if left_child >= 0 || right_child >= 0 {
                nodes.push(TreeNode {
                    id: i,
                    split_index: record.split_features[i],
                    split_condition: NotNan::new(record.thresholds[i]).unwrap(),
                    left: if left_child >= 0 {
                        Some(left_child as usize)
                    } else {
                        None
                    },
                    right: if right_child >= 0 {
                        Some(right_child as usize)
                    } else {
                        None
                    },
                    value: NotNan::new(0.0).unwrap(),
                });
            } else {
                let leaf_index = (-left_child) as usize - 1;
                nodes.push(TreeNode {
                    id: i,
                    split_index: 0,
                    split_condition: NotNan::new(0.0).unwrap(),
                    left: None,
                    right: None,
                    value: NotNan::new(record.left_values[leaf_index]).unwrap(),
                });
            }
        }

        Tree::from_nodes(nodes)
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
        } else if let Some((key, value)) = line.split_once('=')
            && key == "num_tree_per_iteration"
        {
            num_tree_per_iteration = Some(value.parse()?);
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
