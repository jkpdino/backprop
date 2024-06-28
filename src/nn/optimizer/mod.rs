mod sgd;

pub use sgd::*;

use crate::{device::{self, Device}, tensor::TensorRef};

pub trait OptimizerConfig {
    type Optimizer;

    fn build_optimizer(&self, tensors: Vec<TensorRef>, device: Device) -> Self::Optimizer;
}