# Plan: Fix #7 — LightGBM threshold-equality routing bug

Issue: [#7 [BUG][HIGH] Tree prediction uses wrong comparison operator for LightGBM (< should be <=)](https://github.com/lucidfrontier45/silva/issues/7)

## Goal
Make `Tree::predict` route correctly for both XGBoost (`x < threshold → left`) and LightGBM/sklearn (`x <= threshold → left`) by adding a per-tree split-comparison flag set by each parser.

## Background
`src/tree.rs:55` hardcodes `feature < node.split_condition` — XGBoost semantics. LightGBM uses `<=`. At exact IEEE-754 equality (feature == threshold), LightGBM trees route right (wrong leaf) instead of left. Measure-zero for random continuous test features → existing `test_model_prediction` tolerance (0.05) never catches it. `Tree` / `Forest` / `MultiOutputForest` carry no format flag.

Decision: **flag goes on `Tree`** because:
- `Tree::predict` is where the comparison lives.
- `Tree` is public API; `Forest::predict` just delegates.
- All trees in one forest share format, but 1 byte/tree is negligible; avoids touching `Forest` / `MultiOutputForest` signatures.
- Issue explicitly lists "Tree/Forest" — Tree is the minimal/cleanest placement.

## Approach
1. **`src/tree.rs`** — add split-comparison type + field:
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
   #[serde(rename_all = "snake_case")]
   pub enum SplitComparison {
       #[default]
       Less,         // XGBoost:  x <  t → left
       LessOrEqual,  // LightGBM: x <= t → left
   }
   ```
   - Add field to `Tree`: `#[serde(rename = "cmp", default)] split_comparison: SplitComparison`.
   - Keep `Tree::new` / `Tree::from_nodes` signatures unchanged (default `Less`) → **no public API break** for existing users.
   - Add `Tree::new_with_comparison`, `Tree::from_nodes_with_comparison`, and `pub fn split_comparison(&self)`.
   - Rewrite `predict` routing:
     ```rust
     let go_left = match self.split_comparison {
         SplitComparison::Less => feature < node.split_condition,
         SplitComparison::LessOrEqual => feature <= node.split_condition,
     };
     ```
   - Fix struct-literal in existing `test_tree` (adds field) — behavior unchanged, still asserts `<`.

2. **`src/parser/lightgbm.rs`** — `impl From<LGBMTreeRecord> for Tree`: switch `Tree::from_nodes` → `Tree::from_nodes_with_comparison(nodes, SplitComparison::LessOrEqual)`.

3. **`src/parser/xgboost.rs`** — make explicit: `Tree::from_nodes_with_comparison(nodes, SplitComparison::Less)` for documentation clarity (functionally identical to current default).

4. **Tests (the real fix-validation):**
   - `tree.rs`: new `test_predict_less_or_equal` — single split at `sc=5.0`, assert feature `5.0` → left leaf under `LessOrEqual`, → right leaf under `Less`. Locks the divergent behavior at exact equality.
   - `tree.rs`: confirm default `from_nodes` is `Less` (regression guard).
   - `lightgbm.rs`: unit test on `LGBMTreeRecord::into()` asserting `split_comparison() == LessOrEqual` (catches the bug without relying on measure-zero integration data).

5. **Docs (`README.md`):**
   - Tree section: note operator is format-dependent (XGBoost `<`, LightGBM `<=`).
   - Silva Format: `cmp` optional field, default `less`, values `less` / `less_or_equal`.
   - Field Notation table: add `cmp` row.

## Trade-offs
- **Flag on Forest instead** → rejected: `Forest::predict` delegates to `Tree::predict`; would require threading op through every call or a separate predict path. More churn, no benefit.
- **Separate predict path per format** → rejected: duplicates traversal, harder to maintain.
- **bool field `use_le`** → rejected: less self-documenting than enum; enum matches domain language.
- **Touching `default_left` (NaN routing)** → out of scope; issue is strictly the comparison operator. Separate concern.

## Open questions (resolved defaults)
1. **`SplitComparison` visibility** → `pub` on `Tree`, exposed via `pub fn split_comparison(&self)`. `Tree` is already public API; manual tree builders need it.
2. **Serde enum casing** → `snake_case` (`less` / `less_or_equal`). Readable over compact.
3. **`xgboost.rs` explicit `Less` call** → yes, for clarity/symmetry with LightGBM parser.

## Next step
Implement in order: `tree.rs` → `lightgbm.rs` → `xgboost.rs` → tests → `cargo test` → `README.md`.
