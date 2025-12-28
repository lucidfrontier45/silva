# Essential Commands for Silva Development

## Build Commands
- `cargo build` - Build debug version
- `cargo build --release` - Build optimized release version

## Testing Commands
- `cargo test` - Run all tests
- `cargo test --lib` - Run library tests only
- `cargo test <pattern> -- --nocapture` - Run specific test pattern with output
- `cargo test --exact test_name` - Run exact test match

## Code Quality Commands
- `cargo fmt` - Auto-format code
- `cargo fmt --all -- --check` - Check formatting without fixing
- `cargo clippy --all-targets -- -D warnings` - Lint with warnings as errors

## Git Commands
- `git status` - Check working tree status
- `git add .` - Stage all changes
- `git commit -m "message"` - Commit changes
- `git push` - Push to remote
- `git pull` - Pull from remote

## File System Commands
- `ls -la` - List files with details
- `find . -name "*.rs"` - Find Rust source files
- `grep -r "pattern" src/` - Search in source files
- `cargo doc --open` - Generate and open documentation

## Development Workflow
1. Make changes to source code
2. Run `cargo test` to verify functionality
3. Run `cargo fmt` to format code
4. Run `cargo clippy` to check for issues
5. Commit changes with descriptive message

## Performance Testing
- `cargo build --release` - Test optimized build performance
- Use `test_data/` models for integration testing