# Silva

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