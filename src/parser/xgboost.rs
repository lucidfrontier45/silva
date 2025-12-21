use std::path::Path;

use anyhow::{Context, Result as AnyResult};
use itertools::izip;
use serde::{Deserialize, Serialize};
use serdeio::read_record_from_file;

use crate::{Forest, Tree, TreeNode};

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
    pub objective: Objective,
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
                left: if left >= 0 { Some(left as usize) } else { None },
                right: if right >= 0 {
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
#[serde(tag = "name")]
pub enum Objective {
    #[serde(rename = "reg:squarederror")]
    RegSquaredError {
        reg_loss_param: Option<RegLossParamRecord>,
    },
    #[serde(rename = "binary:logistic")]
    BinaryLogistic {
        reg_loss_param: Option<RegLossParamRecord>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegLossParamRecord {
    pub scale_pos_weight: Option<String>,
}

pub fn parse_xgboost_model(record: XGBoostModelRecord) -> Forest {
    let (trees, _tree_info) =
        if let GradientBooster::Gbtree { model } = record.learner.gradient_booster {
            model.parse()
        } else {
            panic!("Only gbtree models are supported");
        };

    let base_score: f64 = record
        .learner
        .learner_model_param
        .base_score
        .parse()
        .unwrap();

    let base_score = match record.learner.objective {
        Objective::RegSquaredError { .. } => base_score,
        Objective::BinaryLogistic { .. } => base_score / (1.0 - base_score),
        _ => panic!("Unsupported objective function"),
    };

    Forest::new(base_score, trees)
}

pub fn read_xgboost_model(path: impl AsRef<Path>) -> AnyResult<Forest> {
    read_record_from_file(path)
        .context("Failed to read XGBoost model file")
        .map(parse_xgboost_model)
}
