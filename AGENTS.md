# Agent guidelines for working in this repo

## Build / Lint / Test Commands
- Build (debug): `cargo build`
- Build (release): `cargo build --release`
- Run all tests: `cargo test`
- Run single test by name: `cargo test <pattern> -- --nocapture` (use `--exact` for exact matches: `cargo test --exact test_name`)
- Run library tests only: `cargo test --lib`
- Format check: `cargo fmt --all -- --check` (auto-fix: `cargo fmt`)
- Lint: `cargo clippy --all-targets -- -D warnings`

## Code Style Guidelines

### General Principles
- This is a Rust machine learning library for tree ensemble models (XGBoost/LightGBM)
- Use Rust 2024 edition idioms throughout
- Follow functional programming patterns where appropriate
- Maintain safety and performance for numerical computations

### Imports and Dependencies
- Group imports in order: `std` -> external crates -> crate-local modules
- Avoid glob imports (`use foo::*`) except for test modules when appropriate
- External crates used: `anyhow`, `indexmap`, `itertools`, `ordered-float`, `rustc-hash`, `serde`, `serde_json`, `serdeio`
- Custom type alias: `FxIndexMap<K, V>` for fast hash maps with deterministic iteration

### Naming Conventions
- Functions and variables: `snake_case`
- Types, structs, enums: `CamelCase`
- Constants: `UPPER_SNAKE_CASE`
- Module names: `snake_case`
- Use descriptive names that reflect ML domain concepts (e.g., `split_condition`, `node_map`)

### Types and Serialization
- Use explicit types on public APIs for clarity
- Use type inference locally when it improves readability
- All model structs must implement `Serialize` and `Deserialize`
- Use `ordered_float::NotNan<f64>` for all floating-point values to prevent NaN issues
- Prefer custom type aliases for frequently used complex types

### Visibility and Encapsulation
- Keep items private by default
- Use `pub(crate)` before `pub` when appropriate for internal sharing
- Expose minimal public API from crate root (`lib.rs`)
- Internal implementation details should be `pub(crate)` at most

### Error Handling
- Use `anyhow::Result<T>` for application boundaries (already in Cargo.toml)
- Use `?` operator for error propagation throughout
- Avoid `unwrap()` and `expect()` in library code
- Use `Context` from anyhow for better error messages
- Only panic on invariant violations that should never occur

### Data Structures
- Use `FxIndexMap` for node mappings in trees (deterministic iteration + fast)
- Use `Option<usize>` for tree node references
- Prefer structured enums for different model types (e.g., `GradientBooster`)
- Use serde field renames for compatibility with external model formats

### Numerical Safety
- Always use `NotNan<f64>` for floating-point values that must be valid
- Use `NotNan::new(value).unwrap()` only for constants/literals
- Avoid implicit floating-point assumptions in tree logic

### Testing Practices
- Write small, deterministic unit tests for all public methods
- Test tree prediction logic with known input/output pairs
- Use table-driven tests when testing multiple variants
- Include integration tests for model parsing from files
- Test error paths and edge cases
- Keep test data in `test_data/` directory with clear organization

### Performance Considerations
- Use efficient data structures for tree traversals
- Avoid unnecessary allocations in prediction loops
- Consider cache-friendly memory layouts for large models
- Use `rustc-hash` for fast hashing when security isn't required

## Serena MCP Server Integration

This repository is configured to work with the Serena MCP server for enhanced code intelligence and automation.

### Serena Tools Available
- **Code Navigation**: `find_symbol`, `find_referencing_symbols`, `get_symbols_overview`
- **Pattern Search**: `search_for_pattern` for flexible code discovery
- **Symbol Manipulation**: `replace_symbol_body`, `insert_before_symbol`, `insert_after_symbol`, `rename_symbol`
- **Memory Management**: `write_memory`, `read_memory`, `list_memories` for project knowledge
- **Project Management**: `activate_project`, `check_onboarding_performed`
- **File Operations**: `find_file`, `list_dir` for project exploration

### Serena Workflow Integration
- Use Serena for symbol discovery before making changes to understand impact
- Leverage memory system to maintain project context and conventions
- Use pattern search to find similar implementations across codebase
- Utilize symbol manipulation tools for safe refactoring
- Check onboarding status when working with new project areas

### Memory System - SECURITY CRITICAL
The project maintains Serena memories in `.serena/memories/`:
- `project_overview.md` - Project structure and purpose
- `suggested_commands.md` - Development commands and workflows  
- `code_style_conventions.md` - Coding standards and patterns
- `task_completion_checklist.md` - Quality verification checklist

**IMPORTANT SECURITY NOTE**: Never include sensitive information in Serena memories:
- No API keys, passwords, tokens, or credentials
- No personal data or identifying information
- No proprietary business logic or trade secrets
- No environment-specific configuration values
- No deployment secrets or production data

Memory should only contain general project structure, coding patterns, and development workflow information.

### Security Guidelines for Serena Usage
- **Memory Content**: Only store general development patterns and project structure
- **No Secrets**: Never store API keys, passwords, tokens, or any credentials in memories
- **Code Reviews**: Ensure no sensitive data accidentally captured in code comments or documentation
- **Git Safety**: Verify `.gitignore` properly excludes sensitive files before committing
- **Environment Separation**: Keep production and development configuration separate
- **Data Sanitization**: Review test data for sensitive information before inclusion

## Agent Behavior Notes
- This repository has no Cursor or Copilot special rules
- Respect module boundaries and visibility rules when making changes
- Focus on correctness and safety in ML model implementations
- Always run `cargo test` after substantial changes
- Preserve backward compatibility with existing model file formats
- Keep test coverage high for core prediction logic
- When adding new model formats, follow existing parser patterns
- **SERENA SECURITY**: Never commit sensitive information (API keys, credentials, personal data) to Serena memories
- **SERENA WORKFLOW**: Use Serena MCP server tools for code analysis and safe refactoring
- **SERENA MEMORY**: Leverage memory system for project context, avoiding sensitive data
