mod sgd;

pub use sgd::*;

use crate::device::Device;

pub trait OptimizerConfig {
    type Optimizer;

    fn build_optimizer(&self, device: &Device) -> Self::Optimizer;
}