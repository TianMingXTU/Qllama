//! CPU backend

use super::Backend;

/// CPU backend
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }
}
