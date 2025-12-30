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