# Silva Project Overview

Silva is a tiny inference engine for tree ensemble models (a.k.a forest models) written in Rust. It provides a lightweight, efficient library for running predictions with pre-trained tree models.

## Purpose
- Fast inference for tree ensemble models
- Support for popular model formats (XGBoost, LightGBM, native Silva format)
- Type-safe numerical computations with NaN protection
- No training capabilities - pure inference engine

## Supported Formats
- Silva format (native)
- XGBoost (regression and classification only)
- LightGBM

## Tech Stack
- **Language**: Rust (edition 2024)
- **Core Dependencies**:
  - `thiserror` - Custom error types
  - `indexmap` - Deterministic iteration hash maps
  - `itertools` - Iterator utilities
  - `ordered-float` - NaN-safe floating point
  - `rustc-hash` - Fast hashing
  - `serde` + `serde_json` - Serialization/deserialization
  - `serdeio` - I/O utilities

## Codebase Structure
```
src/
├── lib.rs              # Public API exports
├── forest.rs           # Forest and MultiOutputForest structs
├── tree.rs             # Tree and TreeNode structs
├── map.rs              # Type aliases for fast maps
└── parser/             # Model format parsers
    ├── mod.rs          # Parser module exports
    ├── builtin.rs      # Native Silva format
    ├── xgboost.rs      # XGBoost format support
    ├── lightgbm.rs     # LightGBM format support
    └── test_utils.rs   # Testing utilities

test_data/              # Sample models and test data
├── lightgbm/          # LightGBM test models
└── xgboost/            # XGBoost test models
```

## Key Concepts
- **Forest**: Collection of trees for single output prediction
- **MultiOutputForest**: Multiple forests for multi-output scenarios
- **Tree**: Single decision tree with nodes
- **TreeNode**: Individual tree nodes with split conditions or leaf values
- **FxIndexMap**: Fast hash map with deterministic iteration order

## Performance Characteristics
- Efficient tree traversal algorithms
- Cache-friendly memory layout
- Minimal allocation during prediction
- NaN-safe numerical operations using NotNan<f64>