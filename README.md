<img src="logo.png" alt="logo" width="300">

Silba is a tiny inference engine for tree ensemble models (a.k.a forest models) in Rust.

# Supported Formats

## Silva Format
- Native format using efficient serde serialization
- Most compact and fastest to load

## XGBoost
- **Booster Types**: `gbtree` only (gblinear and dart are not supported)
- **Supported Objectives**: 
  - `reg:squarederror` (regression)
  - `binary:logistic` (binary classification)
  - `multi:softmax` (multiclass classification)
  - `multi:softprob` (multiclass classification)
- **Note**: Unsupported booster types/objectives will return descriptive errors

## LightGBM
- All regression and classification models
- Text format only (binary format not supported)
- Tree structure only (no linear models)
- Note: LightGBM incorporates all bias into leaf values (no separate base_score)

# Use this library

```sh
cargo add silva
```

# Data Structures

## MultiOutputForest
A container for multi-output models (e.g., multi-class classification). Holds a vector of `Forest` instances, one per output class. Returns a vector of predictions, one per output.

## Forest
Single-output tree ensemble containing:
- `base_value`: Bias/baseline score added to all predictions
- `trees`: Vector of decision trees

Prediction formula: `base_value + Σ tree_predictions`

## Tree
Individual decision tree represented as:
- `node_map`: Hash map of node ID → `TreeNode`
- `root`: Root node ID

Traverses tree from root to leaf based on feature comparisons.

## TreeNode
Single node with:
- `split_index`: Feature index for splitting
- `split_condition`: Threshold value (NotNan<f64>)
- `left/right`: Child node IDs (None for leaves)
- `value`: Leaf value (NotNan<f64>)

Leaves have no children; internal nodes contain split logic.

# Silva Format Example

```json
{
  "forests": [
    {
      "base_value": 0.5,
      "trees": [
        {
          "nm": {
            "0": {"id": 0, "si": 0, "sc": 2.5, "l": 1, "r": 2, "v": 0.0},
            "1": {"id": 1, "si": 1, "sc": 1.5, "l": null, "r": null, "v": 3.0},
            "2": {"id": 2, "si": 1, "sc": 3.5, "l": null, "r": null, "v": 5.0}
          },
          "root": 0
        },
        {
          "nm": {
            "0": {"id": 0, "si": 0, "sc": 5.0, "l": 1, "r": 2, "v": 0.0},
            "1": {"id": 1, "si": 1, "sc": 2.0, "l": null, "r": null, "v": 10.0},
            "2": {"id": 2, "si": 1, "sc": 3.0, "l": null, "r": null, "v": 20.0}
          },
          "root": 0
        }
      ]
    }
  ]
}
```

## Field Notation

| Abbreviation | Full Name | Description |
|--------------|-----------|-------------|
| `nm` | node_map | Hash map mapping node ID to TreeNode |
| `si` | split_index | Feature index used for splitting at this node |
| `sc` | split_condition | Threshold value for the split comparison |
| `l` | left | ID of left child node (null for leaves) |
| `r` | right | ID of right child node (null for leaves) |
| `v` | value | Leaf prediction value (only used in leaf nodes) |

## Structure Hierarchy

```
MultiOutputForest
└── forests: Forest[]
    ├── base_value: f64 (baseline score)
    ├── trees: Tree[]
    │   ├── nm: {node_id: TreeNode}
    │   │   ├── id: node ID
    │   │   ├── si: feature index to split on
    │   │   ├── sc: split threshold
    │   │   ├── l: left child ID (or null)
    │   │   ├── r: right child ID (or null)
    │   │   └── v: leaf value
    │   └── root: ID of the root node
```

**Prediction Flow**: Start at root → compare feature[si] with sc → follow l or r → repeat until leaf → sum all tree values → add base_value

# Usage Examples

## Basic Prediction

The predict methods work with feature vectors (`&[f64]`) and return prediction values.

### Single Tree Prediction
```rust
use silva::Tree;

let tree = Tree::new(node_map, root_id);
let prediction = tree.predict(&[1.5, 2.3, 0.8]); // returns NotNan<f64>
```

### Forest (Single Output)
```rust
use silva::Forest;

let forest = Forest::new(base_value, trees);
let prediction = forest.predict(&[1.5, 2.3, 0.8]); // returns NotNan<f64>
```

### Multi-Output Forest
```rust
use silva::MultiOutputForest;

let model = MultiOutputForest::new(forests);
let predictions = model.predict(&[1.5, 2.3, 0.8]); // returns Vec<NotNan<f64>>
```

## Complete Workflow Example

```rust
use silva::MultiOutputForest;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model from file
    let model = MultiOutputForest::from_file("model.json")?;
    
    // Prepare feature data
    let features = vec![vec![1.5, 2.3, 0.8], vec![0.5, 1.2, 3.4]];
    
    // Make predictions
    for x in &features {
        let prediction = model.predict(x);
        println!("Predictions: {:?}", prediction);
    }
    
    Ok(())
}
```

For more examples, see `examples/prediction.rs`.