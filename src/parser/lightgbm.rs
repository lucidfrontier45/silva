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
    leaf_values: Vec<f64>,
}

impl From<LGBMTreeRecord> for Tree {
    fn from(record: LGBMTreeRecord) -> Self {
        use ordered_float::NotNan;

        let mut nodes = Vec::new();

        let num_internal = record.split_features.len();

        for i in 0..record.split_features.len() {
            let left_child = record.left_children[i];
            let right_child = record.right_children[i];

            let node = TreeNode {
                id: i,
                split_index: record.split_features[i],
                split_condition: NotNan::new(record.thresholds[i]).unwrap(),
                left: if left_child >= 0 {
                    Some(left_child as usize)
                } else {
                    let leaf_id = num_internal + ((-left_child) as usize - 1);
                    Some(leaf_id)
                },
                right: if right_child >= 0 {
                    Some(right_child as usize)
                } else {
                    let leaf_id = num_internal + ((-right_child) as usize - 1);
                    Some(leaf_id)
                },
                value: NotNan::new(0.0).unwrap(),
            };
            nodes.push(node);
        }

        for (i, &leaf_value) in record.leaf_values.iter().enumerate() {
            let leaf_id = num_internal + i;
            let leaf_node = TreeNode {
                id: leaf_id,
                split_index: 0,
                split_condition: NotNan::new(0.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(leaf_value).unwrap(),
            };
            nodes.push(leaf_node);
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
        leaf_values: leaf_values?,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::parser::{read_lightgbm_model, test_utils::test_model_prediction};

    fn test_lightgbm(model_type: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let root = PathBuf::from(manifest_dir);
        let data_dir = root.join(format!("test_data/lightgbm/{}", model_type));
        let model_path = data_dir.join("model.txt");
        let forest = read_lightgbm_model(&model_path).expect("Failed to load model");

        test_model_prediction(&data_dir, &forest, 0.05)
            .unwrap_or_else(|e| panic!("LightGBM {model_type} model prediction test failed: {e}"));
    }

    #[test]
    fn test_regression() {
        test_lightgbm("regression");
    }

    #[test]
    fn test_binary_classification() {
        test_lightgbm("binary_classification");
    }

    #[test]
    fn test_multiclass_classification() {
        test_lightgbm("multiclass_classification");
    }
}
