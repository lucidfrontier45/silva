# Silva Code Style Conventions

## General Principles
- Rust 2024 edition idioms throughout
- Functional programming patterns where appropriate
- Safety and performance prioritized for numerical computations
- ML domain-specific naming conventions

## Import Organization
Order: `std` -> external crates -> crate-local modules
```rust
use std::collections::HashMap;
use thiserror::Error;
use crate::tree::Tree;
```

## Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `split_condition`, `node_map`)
- **Types/Structs/Enums**: `CamelCase` (e.g., `TreeNode`, `GradientBooster`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_DEPTH`)
- **Modules**: `snake_case` (e.g., `parser`, `forest`)

## Type System Guidelines
- Use `ordered_float::NotNan<f64>` for all floating-point values
- Explicit types on public APIs, type inference locally
- Custom type alias: `FxIndexMap<K, V>` for deterministic iteration
- All model structs must implement `Serialize` and `Deserialize`

## Visibility Rules
- Keep items private by default
- Use `pub(crate)` before `pub` for internal sharing
- Expose minimal public API from `lib.rs`
- Internal implementation: `pub(crate)` at most

## Error Handling
- Use custom error types with `thiserror` for library code
- Propagate errors with `?` operator
- Avoid `unwrap()` and `expect()` in library code
- Use `anyhow` only in test/example code
- Only panic on invariant violations

## Data Structures
- `FxIndexMap` for tree node mappings (deterministic + fast)
- `Option<usize>` for tree node references
- Structured enums for different model types
- Serde field renames for external format compatibility

## Numerical Safety
- Always use `NotNan<f64>` for floating-point values that must be valid
- Use `NotNan::new(value).unwrap()` only for constants/literals
- Avoid implicit floating-point assumptions in tree logic

## Testing Style
- Small, deterministic unit tests for all public methods
- Known input/output pairs for prediction logic
- Table-driven tests for multiple variants
- Integration tests with files from `test_data/`
- Test error paths and edge cases

## Performance Guidelines
- Efficient data structures for tree traversals
- Minimize allocations in prediction loops
- Cache-friendly memory layouts for large models
- Use `rustc-hash` for fast hashing when security isn't required

## Serde Patterns
- Use field renames for compatibility: `#[serde(rename(serialize = "si", deserialize = "si"))]`
- Compact field names in serialized format
- Maintain backward compatibility with existing model files