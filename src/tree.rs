use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use crate::map::FxIndexMap;

/// Comparison operator used at split nodes.
///
/// Different source frameworks use different conventions:
/// - XGBoost: `x < threshold` routes to the left child.
/// - LightGBM (and scikit-learn): `x <= threshold` routes to the left child.
///
/// A single hardcoded operator cannot satisfy both formats, so each parser sets this
/// flag on the trees it produces. The default (`Less`) matches the XGBoost convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SplitComparison {
    /// `x < threshold` routes to the left child. Used by XGBoost.
    #[default]
    Less,
    /// `x <= threshold` routes to the left child. Used by LightGBM and scikit-learn.
    LessOrEqual,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TreeNode {
    pub(crate) id: usize,
    #[serde(rename(serialize = "si", deserialize = "si"))]
    pub(crate) split_index: usize,
    #[serde(rename(serialize = "sc", deserialize = "sc"))]
    pub(crate) split_condition: NotNan<f64>,
    #[serde(rename(serialize = "l", deserialize = "l"))]
    pub(crate) left: Option<usize>,
    #[serde(rename(serialize = "r", deserialize = "r"))]
    pub(crate) right: Option<usize>,
    #[serde(rename(serialize = "v", deserialize = "v"))]
    pub(crate) value: NotNan<f64>,
}

impl TreeNode {
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    pub fn get_value(&self) -> NotNan<f64> {
        self.value
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    #[serde(rename(serialize = "nm", deserialize = "nm"))]
    node_map: FxIndexMap<usize, TreeNode>,
    root: usize,
    /// Comparison operator applied at split nodes. XGBoost uses `Less`, LightGBM uses `LessOrEqual`.
    /// Defaults to `Less` via `#[serde(default)]` so older Silva-format files deserialize unchanged.
    #[serde(rename = "cmp", default)]
    split_comparison: SplitComparison,
}

impl Tree {
    pub fn new(node_map: FxIndexMap<usize, TreeNode>, root: usize) -> Self {
        Self {
            node_map,
            root,
            split_comparison: SplitComparison::default(),
        }
    }

    pub fn from_nodes(mut nodes: Vec<TreeNode>) -> Self {
        nodes.sort_by_key(|node| node.id);
        let root_id = nodes[0].id;
        let node_map: FxIndexMap<usize, TreeNode> =
            nodes.into_iter().map(|node| (node.id, node)).collect();
        Self::new(node_map, root_id)
    }

    /// Construct a tree with an explicit comparison operator.
    ///
    /// Used by the LightGBM parser (`LessOrEqual`) and by the XGBoost parser for
    /// symmetry (`Less`). Manual tree builders can use this to embed a different
    /// operator convention (e.g. scikit-learn also uses `LessOrEqual`).
    pub fn new_with_comparison(
        node_map: FxIndexMap<usize, TreeNode>,
        root: usize,
        split_comparison: SplitComparison,
    ) -> Self {
        Self {
            node_map,
            root,
            split_comparison,
        }
    }

    /// Like [`Tree::from_nodes`] but with an explicit comparison operator.
    pub fn from_nodes_with_comparison(
        mut nodes: Vec<TreeNode>,
        split_comparison: SplitComparison,
    ) -> Self {
        nodes.sort_by_key(|node| node.id);
        let root_id = nodes[0].id;
        let node_map: FxIndexMap<usize, TreeNode> =
            nodes.into_iter().map(|node| (node.id, node)).collect();
        Self::new_with_comparison(node_map, root_id, split_comparison)
    }

    /// Returns the comparison operator this tree uses at split nodes.
    pub fn split_comparison(&self) -> SplitComparison {
        self.split_comparison
    }

    pub fn predict(&self, x: &[f64]) -> NotNan<f64> {
        let mut node = self.node_map.get(&self.root).unwrap();
        while !node.is_leaf() {
            let feature = NotNan::new(x[node.split_index]).unwrap();
            let go_left = match self.split_comparison {
                SplitComparison::Less => feature < node.split_condition,
                SplitComparison::LessOrEqual => feature <= node.split_condition,
            };
            let next_node = if go_left { node.left } else { node.right }
                .and_then(|id| self.node_map.get(&id))
                .unwrap();
            node = next_node;
        }
        node.get_value()
    }
}

#[cfg(test)]
mod test {
    use ordered_float::NotNan;

    use crate::{
        map::FxIndexMap,
        tree::{SplitComparison, Tree, TreeNode},
    };

    #[test]
    fn test_tree() {
        // Build nodes
        let mut nodes = FxIndexMap::default();
        nodes.insert(
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
        nodes.insert(
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
        nodes.insert(
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
        nodes.insert(
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
        nodes.insert(
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
        nodes.insert(
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
        nodes.insert(
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

        let tree = Tree {
            node_map: nodes,
            root: 0,
            split_comparison: SplitComparison::Less,
        };

        assert_eq!(tree.predict(&[4.0, 2.0]), NotNan::new(3.0).unwrap());
        assert_eq!(tree.predict(&[4.0, 4.0]), NotNan::new(4.0).unwrap());
        assert_eq!(tree.predict(&[6.0, 1.0]), NotNan::new(5.0).unwrap());
        assert_eq!(tree.predict(&[6.0, 3.0]), NotNan::new(6.0).unwrap());
    }

    /// Build a single-split tree with threshold 5.0: feature<5 -> leaf value 10, else leaf value 20.
    fn two_leaf_tree(split_comparison: SplitComparison) -> Tree {
        let mut nodes = FxIndexMap::default();
        nodes.insert(
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
        nodes.insert(
            1,
            TreeNode {
                id: 1,
                split_index: 0,
                split_condition: NotNan::new(0.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(10.0).unwrap(),
            },
        );
        nodes.insert(
            2,
            TreeNode {
                id: 2,
                split_index: 0,
                split_condition: NotNan::new(0.0).unwrap(),
                left: None,
                right: None,
                value: NotNan::new(20.0).unwrap(),
            },
        );
        Tree::new_with_comparison(nodes, 0, split_comparison)
    }

    #[test]
    fn test_predict_default_is_less() {
        // `Tree::from_nodes` (and `Tree::new`) default to the XGBoost convention.
        let nodes = vec![TreeNode {
            id: 0,
            split_index: 0,
            split_condition: NotNan::new(0.0).unwrap(),
            left: None,
            right: None,
            value: NotNan::new(0.0).unwrap(),
        }];
        assert_eq!(
            Tree::from_nodes(nodes).split_comparison(),
            SplitComparison::Less
        );
    }

    #[test]
    fn test_predict_at_threshold_equality() {
        // Regression test for issue #7: at exact equality between feature and threshold,
        // XGBoost (`<`) routes to the right leaf, LightGBM (`<=`) routes to the left leaf.
        let less = two_leaf_tree(SplitComparison::Less);
        let less_or_equal = two_leaf_tree(SplitComparison::LessOrEqual);

        // Strictly below: both route left.
        assert_eq!(less.predict(&[4.99]), NotNan::new(10.0).unwrap());
        assert_eq!(less_or_equal.predict(&[4.99]), NotNan::new(10.0).unwrap());
        // Strictly above: both route right.
        assert_eq!(less.predict(&[5.01]), NotNan::new(20.0).unwrap());
        assert_eq!(less_or_equal.predict(&[5.01]), NotNan::new(20.0).unwrap());
        // Exact equality: the divergence that issue #7 is about.
        assert_eq!(less.predict(&[5.0]), NotNan::new(20.0).unwrap());
        assert_eq!(less_or_equal.predict(&[5.0]), NotNan::new(10.0).unwrap());
    }
}
