use super::OptimizerConfig;

///
/// A OptimizerConfig for Stochastic Gradient Descent.
/// 
#[derive(Clone, PartialEq)]
pub struct SgdConfig {
    pub lr: f32,
}

pub struct Sgd {
    cfg: SgdConfig
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

    fn build_optimizer(&self, _device: &crate::device::Device) -> Sgd {
        Sgd { cfg: self.clone() }
    }
}

impl Sgd {
    pub fn update(
        &self,
    ) {

    }
}