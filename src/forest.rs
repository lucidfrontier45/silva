use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use crate::tree::Tree;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forest {
    base_value: f64,
    trees: Vec<Tree>,
}

impl Forest {
    pub fn new(base_value: f64, trees: Vec<Tree>) -> Self {
        Self { base_value, trees }
    }

    pub fn predict(&self, x: &[f64]) -> NotNan<f64> {
        let predictions: Vec<f64> = self
            .trees
            .iter()
            .map(|tree| tree.predict(x).into_inner())
            .collect();

        let res = self.base_value + predictions.iter().sum::<f64>();

        NotNan::new(res).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use crate::{
        map::FxIndexMap,
        tree::{Tree, TreeNode},
    };

    use super::*;

    #[test]
    fn test_forest_predict() {
        // First tree
        let mut nodes1 = FxIndexMap::default();
        nodes1.insert(
            0,
            TreeNode {
                id: 0,
                split_index: 0,
                split_condition: NotNan::new(5.0).unwrap(),
                left: Some(1),
                right: Some(2),
                value: NotNan::new(0.0).unwrap(),
            },
        );
        nodes1.insert(
            1,
            TreeNode {
                id: 1,
                split_index: 1,
                split_condition: NotNan::new(3.0).unwrap(),
                left: Some(3),
                right: Some(4),
                value: NotNan::new(0.0).unwrap(),
            },
        );
        nodes1.insert(
            2,
            TreeNode {
                id: 2,
                split_index: 1,
                split_condition: NotNan::new(2.0).unwrap(),
                left: Some(5),
                right: Some(6),
                value: NotNan::new(0.0).unwrap(),
            },
        );
        nodes1.insert(
            3,
            TreeNode {
                id: 3,
                split_index: 0,
                split_condition: NotNan::new(0.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(3.0).unwrap(),
            },
        );
        nodes1.insert(
            4,
            TreeNode {
                id: 4,
                split_index: 0,
                split_condition: NotNan::new(0.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(4.0).unwrap(),
            },
        );
        nodes1.insert(
            5,
            TreeNode {
                id: 5,
                split_index: 0,
                split_condition: NotNan::new(0.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(5.0).unwrap(),
            },
        );
        nodes1.insert(
            6,
            TreeNode {
                id: 6,
                split_index: 0,
                split_condition: NotNan::new(0.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(6.0).unwrap(),
            },
        );
        let tree1 = Tree::new(nodes1, 0);

        // Second tree
        let mut nodes2 = FxIndexMap::default();
        nodes2.insert(
            0,
            TreeNode {
                id: 0,
                split_index: 0,
                split_condition: NotNan::new(5.0).unwrap(),
                left: Some(1),
                right: Some(2),
                value: NotNan::new(0.0).unwrap(),
            },
        );
        nodes2.insert(
            1,
            TreeNode {
                id: 1,
                split_index: 0,
                split_condition: NotNan::new(2.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(10.0).unwrap(),
            },
        );
        nodes2.insert(
            2,
            TreeNode {
                id: 2,
                split_index: 0,
                split_condition: NotNan::new(3.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(20.0).unwrap(),
            },
        );
        let tree2 = Tree::new(nodes2, 0);

        let forest = Forest::new(100.0, vec![tree1, tree2]);

        assert_eq!(forest.predict(&[4.0, 2.0]), NotNan::new(113.0).unwrap());
        assert_eq!(forest.predict(&[4.0, 4.0]), NotNan::new(114.0).unwrap());
        assert_eq!(forest.predict(&[6.0, 1.0]), NotNan::new(125.0).unwrap());
        assert_eq!(forest.predict(&[6.0, 3.0]), NotNan::new(126.0).unwrap());
    }
}
