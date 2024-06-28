use crate::{device::Device, tensor::{inner::TensorInner, TensorRef}};

use super::OptimizerConfig;

///
/// A OptimizerConfig for Stochastic Gradient Descent.
/// 
#[derive(Clone, PartialEq)]
pub struct SgdConfig {
    pub lr: f32,
}

pub struct Sgd {
    cfg:     SgdConfig,
    device:  Device,
    tensors: Vec<TensorRef>,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self { 
            lr: 0.001
        }
    }
}

impl OptimizerConfig for SgdConfig {
    type Optimizer = Sgd;

    fn build_optimizer(&self, tensors: Vec<TensorRef>, device: Device) -> Sgd {
        Sgd {
            tensors,
            device,
            cfg: self.clone()
        }
    }
}

impl Sgd {
    ///
    /// Resets all gradients
    /// 
    pub fn zero_grad(&mut self) {
        self.device.zero_grad();
    }

    pub fn step(&mut self) {
        for tensor in &self.tensors {
            let buffer = tensor.buffer_mut();
            let gradient = tensor.gradient();

            for (param, gradient) in buffer.iter_mut().zip(gradient) {
                *param -= gradient * self.cfg.lr;
            }
        }
    }
}