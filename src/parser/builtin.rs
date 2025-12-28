use std::path::Path;

use serdeio::read_record_from_file;

use crate::{Forest, MultiOutputForest};

impl Forest {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, serdeio::Error> {
        read_record_from_file(path)
    }
}

impl MultiOutputForest {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, serdeio::Error> {
        read_record_from_file(path)
    }
}
