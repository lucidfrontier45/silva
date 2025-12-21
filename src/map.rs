use indexmap::IndexMap;
use rustc_hash::FxBuildHasher;

pub type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;
