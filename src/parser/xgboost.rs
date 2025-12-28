use std::{collections::HashMap, path::Path, vec};

use itertools::izip;
use serde::{Deserialize, Serialize};
use serdeio::read_record_from_file;
use thiserror::Error;

use crate::{Forest, MultiOutputForest, Tree, TreeNode};

/// Custom error types for XGBoost model parsing
#[derive(Debug, Error)]
pub enum XGBoostError {
    #[error("Unsupported booster type: {booster}. Only 'gbtree' is supported")]
    UnsupportedBooster { booster: String },
    #[error("Unsupported objective function: {objective}. Supported objectives: {supported:?}")]
    UnsupportedObjective {
        objective: String,
        supported: Vec<String>,
    },
    #[error("Invalid base_score format: {value}. Expected format like '0.5' or '[0.1,0.2,0.3]'")]
    InvalidBaseScore { value: String },
    #[error("Model parameter error: {parameter}")]
    InvalidParameters { parameter: String },
    #[error("File read error: {source}")]
    FileRead {
        #[from]
        source: serdeio::Error,
    },
}

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

pub fn parse_xgboost_model(record: XGBoostModelRecord) -> Result<MultiOutputForest, XGBoostError> {
    let (trees, tree_info) = match record.learner.gradient_booster {
        GradientBooster::Gbtree { model } => model.parse(),
        GradientBooster::Gblinear { .. } => {
            return Err(XGBoostError::UnsupportedBooster {
                booster: "gblinear".to_string(),
            });
        }
        GradientBooster::Dart { .. } => {
            return Err(XGBoostError::UnsupportedBooster {
                booster: "dart".to_string(),
            });
        }
    };

    let objective_name = record.learner.objective.name.clone();
    let objective = Objective::from(record.learner.objective);
    // Handle unsupported objectives early to avoid closure issues
    if let Objective::Unknown = objective {
        return Err(XGBoostError::UnsupportedObjective {
            objective: objective_name,
            supported: vec![
                "reg:squarederror".to_string(),
                "binary:logistic".to_string(),
                "multi:softmax".to_string(),
                "multi:softprob".to_string(),
            ],
        });
    }

    // group trees into forests based on tree_info values
    let n_classes = tree_info.iter().max().unwrap() + 1;
    let mut tree_groups = vec![Vec::new(); n_classes];
    for (tree, &class_idx) in trees.into_iter().zip(tree_info.iter()) {
        tree_groups[class_idx].push(tree);
    }

    let base_scores: Vec<f64> = parse_base_score(&record.learner.learner_model_param.base_score)
        .map_err(|e| XGBoostError::InvalidBaseScore { value: e })?;

    let mut base_values: Vec<f64> = base_scores
        .into_iter()
        .map(|s| match objective {
            Objective::RegSquaredError => s,
            Objective::BinaryLogistic => logit(s),
            Objective::MultiSoftmax | Objective::MultiSoftprob => s,
            Objective::Unknown => unreachable!(), // We already handled this above
        })
        .collect();
    if base_values.len() == 1 && n_classes > 1 {
        base_values = vec![base_values[0]; n_classes];
    }

    let forests: Vec<Forest> = tree_groups
        .into_iter()
        .zip(base_values)
        .map(|(trees, base_value)| Forest::new(base_value, trees))
        .collect();

    Ok(MultiOutputForest::new(forests))
}

pub fn read_xgboost_model(path: impl AsRef<Path>) -> Result<MultiOutputForest, XGBoostError> {
    let record = read_record_from_file(path)?;
    parse_xgboost_model(record)
}

fn parse_base_score(s: &str) -> Result<Vec<f64>, String> {
    // [0.1,0.2,.03] -> Vec<f64>
    s.trim_matches(&['[', ']'][..])
        .split(',')
        .map(|x| x.parse::<f64>().map_err(|_| s.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::parser::{read_xgboost_model, test_utils::test_model_prediction};

    fn test_xgboost(model_type: &str) {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let root = PathBuf::from(manifest_dir);
        let data_dir = root.join(format!("test_data/xgboost/{}", model_type));
        let model_path = data_dir.join("model.json");
        let forest = read_xgboost_model(&model_path).expect("Failed to load model");

        test_model_prediction(&data_dir, &forest, 0.05).unwrap_or_else(|e| {
            panic!(
                "XGBoost model prediction test failed for `{}` using data dir {:?}: {}",
                model_type, data_dir, e
            )
        });
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

    // Error handling tests
    #[test]
    fn test_parse_xgboost_model_unsupported_booster_gblinear() {
        let model = XGBoostModelRecord {
            version: [1, 0, 0],
            learner: LearnerRecord {
                feature_names: None,
                feature_types: None,
                gradient_booster: GradientBooster::Gblinear {
                    model: GblinearModelRecord {
                        weights: vec![0.1, 0.2],
                    },
                },
                objective: ObjectiveRecord {
                    name: "reg:squarederror".to_string(),
                    extra_fields: HashMap::new(),
                },
                learner_model_param: LearnerModelParamRecord {
                    base_score: "0.5".to_string(),
                    num_class: None,
                    num_feature: None,
                    num_target: None,
                },
            },
        };

        let result = parse_xgboost_model(model);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("gblinear"));
        assert!(error_msg.contains("gbtree"));
    }

    #[test]
    fn test_parse_xgboost_model_unsupported_booster_dart() {
        let model = XGBoostModelRecord {
            version: [1, 0, 0],
            learner: LearnerRecord {
                feature_names: None,
                feature_types: None,
                gradient_booster: GradientBooster::Dart {
                    gbtree: Box::new(GradientBooster::Gbtree {
                        model: GbtreeModelRecord {
                            gbtree_model_param: GbtreeModelParamRecord {
                                num_trees: "1".to_string(),
                                num_parallel_tree: "1".to_string(),
                            },
                            trees: vec![],
                            tree_info: vec![0],
                        },
                    }),
                    weight_drop: vec![0.1],
                },
                objective: ObjectiveRecord {
                    name: "reg:squarederror".to_string(),
                    extra_fields: HashMap::new(),
                },
                learner_model_param: LearnerModelParamRecord {
                    base_score: "0.5".to_string(),
                    num_class: None,
                    num_feature: None,
                    num_target: None,
                },
            },
        };

        let result = parse_xgboost_model(model);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("dart"));
        assert!(error_msg.contains("gbtree"));
    }

    #[test]
    fn test_parse_xgboost_model_unsupported_objective() {
        let model = XGBoostModelRecord {
            version: [1, 0, 0],
            learner: LearnerRecord {
                feature_names: None,
                feature_types: None,
                gradient_booster: GradientBooster::Gbtree {
                    model: GbtreeModelRecord {
                        gbtree_model_param: GbtreeModelParamRecord {
                            num_trees: "1".to_string(),
                            num_parallel_tree: "1".to_string(),
                        },
                        trees: vec![],
                        tree_info: vec![0],
                    },
                },
                objective: ObjectiveRecord {
                    name: "binary:logitraw".to_string(), // Unsupported objective
                    extra_fields: HashMap::new(),
                },
                learner_model_param: LearnerModelParamRecord {
                    base_score: "0.5".to_string(),
                    num_class: None,
                    num_feature: None,
                    num_target: None,
                },
            },
        };

        let result = parse_xgboost_model(model);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("binary:logitraw"));
        assert!(error_msg.contains("reg:squarederror"));
        assert!(error_msg.contains("binary:logistic"));
        assert!(error_msg.contains("multi:softmax"));
        assert!(error_msg.contains("multi:softprob"));
    }

    #[test]
    fn test_parse_xgboost_model_invalid_base_score() {
        let model = XGBoostModelRecord {
            version: [1, 0, 0],
            learner: LearnerRecord {
                feature_names: None,
                feature_types: None,
                gradient_booster: GradientBooster::Gbtree {
                    model: GbtreeModelRecord {
                        gbtree_model_param: GbtreeModelParamRecord {
                            num_trees: "1".to_string(),
                            num_parallel_tree: "1".to_string(),
                        },
                        trees: vec![],
                        tree_info: vec![0],
                    },
                },
                objective: ObjectiveRecord {
                    name: "reg:squarederror".to_string(),
                    extra_fields: HashMap::new(),
                },
                learner_model_param: LearnerModelParamRecord {
                    base_score: "invalid".to_string(),
                    num_class: None,
                    num_feature: None,
                    num_target: None,
                },
            },
        };

        let result = parse_xgboost_model(model);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("base_score"));
        assert!(error_msg.contains("invalid"));
    }

    #[test]
    fn test_parse_base_score_valid_formats() {
        // Test single value
        let result = parse_base_score("0.5").unwrap();
        assert_eq!(result, vec![0.5]);

        // Test array format
        let result = parse_base_score("[0.1,0.2,0.3]").unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3]);

        // Test single value in brackets
        let result = parse_base_score("[0.7]").unwrap();
        assert_eq!(result, vec![0.7]);
    }

    #[test]
    fn test_parse_base_score_invalid_format() {
        // Test invalid number
        let result = parse_base_score("invalid");
        assert!(result.is_err());

        // Test invalid array
        let result = parse_base_score("[0.1,invalid,0.3]");
        assert!(result.is_err());
    }
}
