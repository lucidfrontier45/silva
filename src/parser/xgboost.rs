use std::{collections::HashMap, path::Path, vec};

use anyhow::{Context, Result as AnyResult};
use itertools::{Itertools, izip};
use serde::{Deserialize, Serialize};
use serdeio::read_record_from_file;

use crate::{Forest, MultiOutputForest, Tree, TreeNode};

#[derive(Debug, Serialize, Deserialize)]
pub struct XGBoostModelRecord {
    pub version: [u32; 3],
    pub learner: LearnerRecord,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LearnerRecord {
    pub feature_names: Option<Vec<String>>,
    pub feature_types: Option<Vec<String>>,
    pub gradient_booster: GradientBooster,
    pub objective: ObjectiveRecord,
    pub learner_model_param: LearnerModelParamRecord,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "name")]
pub enum GradientBooster {
    #[serde(rename = "gbtree")]
    Gbtree { model: GbtreeModelRecord },
    #[serde(rename = "gblinear")]
    Gblinear { model: GblinearModelRecord },
    #[serde(rename = "dart")]
    Dart {
        gbtree: Box<GradientBooster>,
        weight_drop: Vec<f64>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GbtreeModelRecord {
    pub gbtree_model_param: GbtreeModelParamRecord,
    pub trees: Vec<TreeRecord>,
    pub tree_info: Vec<usize>,
}

impl GbtreeModelRecord {
    pub fn parse(self) -> (Vec<Tree>, Vec<usize>) {
        let trees = self.trees.into_iter().map(|tree| tree.parse()).collect();
        let tree_info = self.tree_info;
        (trees, tree_info)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GbtreeModelParamRecord {
    pub num_trees: String,
    pub num_parallel_tree: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TreeRecord {
    pub tree_param: TreeParamRecord,
    pub id: i32,
    pub loss_changes: Vec<f64>,
    pub sum_hessian: Vec<f64>,
    pub base_weights: Vec<f64>,
    pub left_children: Vec<i32>,
    pub right_children: Vec<i32>,
    pub parents: Vec<i32>,
    pub split_indices: Vec<i32>,
    pub split_conditions: Vec<f64>,
    pub split_type: Vec<i32>,
    pub default_left: Vec<i32>,
    pub categories: Vec<i32>,
    pub categories_nodes: Vec<i32>,
    pub categories_segments: Vec<i32>,
    pub categories_sizes: Vec<i32>,
}

impl TreeRecord {
    pub fn parse(self) -> Tree {
        let mut nodes = Vec::new();
        for (i, (_value, left, right, split_index, split_condition)) in izip!(
            self.base_weights,
            self.left_children,
            self.right_children,
            self.split_indices,
            self.split_conditions
        )
        .enumerate()
        {
            let node = TreeNode {
                id: i,
                split_index: split_index as usize,
                split_condition: ordered_float::NotNan::new(split_condition).unwrap(),
                left: if left > 0 { Some(left as usize) } else { None },
                right: if right > 0 {
                    Some(right as usize)
                } else {
                    None
                },
                // surprisingly, the leaf value is taken from split_conditions, not base_weights
                // check https://github.com/dmlc/xgboost/issues/11521
                value: ordered_float::NotNan::new(split_condition).unwrap(),
            };

            nodes.push(node);
        }

        Tree::from_nodes(nodes)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TreeParamRecord {
    pub num_nodes: String,
    pub size_leaf_vector: String,
    pub num_feature: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GblinearModelRecord {
    pub weights: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LearnerModelParamRecord {
    pub base_score: String,
    pub num_class: Option<String>,
    pub num_feature: Option<String>,
    pub num_target: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ObjectiveRecord {
    name: String,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Objective {
    RegSquaredError,
    BinaryLogistic,
    MultiSoftmax,
    MultiSoftprob,
    Unknown,
}

impl From<ObjectiveRecord> for Objective {
    fn from(record: ObjectiveRecord) -> Self {
        match record.name.as_str() {
            "reg:squarederror" => Objective::RegSquaredError,
            "binary:logistic" => Objective::BinaryLogistic,
            "multi:softmax" => Objective::MultiSoftmax,
            "multi:softprob" => Objective::MultiSoftprob,
            _ => Objective::Unknown,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegLossParamRecord {
    pub scale_pos_weight: Option<String>,
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

pub fn parse_xgboost_model(record: XGBoostModelRecord) -> MultiOutputForest {
    let (trees, tree_info) =
        if let GradientBooster::Gbtree { model } = record.learner.gradient_booster {
            model.parse()
        } else {
            panic!("Only gbtree models are supported");
        };

    // group trees into forests based on tree_info values
    let n_classes = tree_info.iter().max().unwrap() + 1;
    let mut tree_groups = vec![Vec::new(); n_classes];
    for (tree, &class_idx) in trees.into_iter().zip(tree_info.iter()) {
        tree_groups[class_idx].push(tree);
    }

    let objective = Objective::from(record.learner.objective);

    let base_scores: Vec<f64> = parse_base_score(&record.learner.learner_model_param.base_score);
    let mut base_values = base_scores
        .into_iter()
        .map(|s| match objective {
            Objective::RegSquaredError => s,
            Objective::BinaryLogistic => logit(s),
            Objective::MultiSoftmax | Objective::MultiSoftprob => s,
            _ => panic!("Unsupported objective function"),
        })
        .collect_vec();
    if base_values.len() == 1 && n_classes > 1 {
        base_values = vec![base_values[0]; n_classes];
    }

    let forests: Vec<Forest> = tree_groups
        .into_iter()
        .zip(base_values)
        .map(|(trees, base_value)| Forest::new(base_value, trees))
        .collect();

    MultiOutputForest::new(forests)
}

pub fn read_xgboost_model(path: impl AsRef<Path>) -> AnyResult<MultiOutputForest> {
    read_record_from_file(path)
        .context("Failed to read XGBoost model file")
        .map(parse_xgboost_model)
}

fn parse_base_score(s: &str) -> Vec<f64> {
    // [0.1,0.2,.03] -> Vec<f64>
    s.trim_matches(&['[', ']'][..])
        .split(',')
        .map(|x| x.parse::<f64>().unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::parser::{read_xgboost_model, test_utils::test_model_prediction};

    fn test_xgboost(model_type: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let root = PathBuf::from(manifest_dir);
        let data_dir = root.join(format!("test_data/xgboost/{}", model_type));
        let model_path = data_dir.join("model.json");
        let forest = read_xgboost_model(&model_path).expect("Failed to load model");

        test_model_prediction(&data_dir, &forest, 0.05).expect("Test failed");
    }

    #[test]
    fn test_regression() {
        test_xgboost("regression");
    }

    #[test]
    fn test_binary_classification() {
        test_xgboost("binary_classification");
    }

    #[test]
    fn test_multiclass_classification() {
        test_xgboost("multiclass_classification");
    }
}
