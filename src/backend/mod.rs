//! Compute backend abstraction

mod cpu;

pub use cpu::CpuBackend;

/// Backend trait
pub trait Backend {
    /// Backend name
    fn name(&self) -> &str;
}
