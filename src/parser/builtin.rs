use std::path::Path;

use anyhow::Result as AnyResult;
use serdeio::read_record_from_file;

use crate::Forest;

impl Forest {
    pub fn from_file(path: impl AsRef<Path>) -> AnyResult<Self> {
        read_record_from_file(path)
    }
}
