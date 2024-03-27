mod relu;
mod softmax;

pub use relu::ReLU;
pub use softmax::Softmax;

use crate::layers::Layer;

pub trait Activation: Sized {
    fn forward(&self, x: &[f32], y: &mut [f32]);

    fn into_layer(self, size: usize) -> ActivationLayer<Self> {
        ActivationLayer (self, size)
    }
}

pub struct ActivationLayer<T: Activation> (T, usize);

impl<T: Activation> Layer for ActivationLayer<T> {
    fn input_size(&self) -> usize {
        self.1
    }

    fn parameters_size(&self) -> usize {
        0
    }

    fn output_size(&self) -> usize {
        self.1
    }

    fn forward(
        &self,
        _parameters: &[f32],
        input_activations: &[f32],
        output_activations: &mut [f32])
    {
        self.0.forward(input_activations, output_activations);
    }
}