use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

use crate::map::FxIndexMap;

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
}

impl Tree {
    pub fn new(node_map: FxIndexMap<usize, TreeNode>, root: usize) -> Self {
        Self { node_map, root }
    }

    pub fn from_nodes(mut nodes: Vec<TreeNode>) -> Self {
        nodes.sort_by_key(|node| node.id);
        let root_id = nodes[0].id;
        let node_map: FxIndexMap<usize, TreeNode> =
            nodes.into_iter().map(|node| (node.id, node)).collect();
        Self::new(node_map, root_id)
    }

    pub fn predict(&self, x: &[f64]) -> NotNan<f64> {
        let mut node = self.node_map.get(&self.root).unwrap();
        while !node.is_leaf() {
            let feature = NotNan::new(x[node.split_index]).unwrap();
            let next_node = if feature < node.split_condition {
                node.left
            } else {
                node.right
            }
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
        tree::{Tree, TreeNode},
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
        };

        assert_eq!(tree.predict(&[4.0, 2.0]), NotNan::new(3.0).unwrap());
        assert_eq!(tree.predict(&[4.0, 4.0]), NotNan::new(4.0).unwrap());
        assert_eq!(tree.predict(&[6.0, 1.0]), NotNan::new(5.0).unwrap());
        assert_eq!(tree.predict(&[6.0, 3.0]), NotNan::new(6.0).unwrap());
    }
}
