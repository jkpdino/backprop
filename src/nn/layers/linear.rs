use crate::{device::Device, nn::Activation, tensor::{inner::TensorInner, Rank1, Rank2, Tensor, TensorRef}, tensor_ops::matmul};

use super::{Layer, LayerBuilder};

///
/// A linear, dense, or fully connected layer.
/// 
/// This layer applies a linear transformation to the input tensor, followed by an activation function.
/// 
/// The layer is parameterized by the input and output shapes, and the activation function.
/// 
pub struct Linear<const I: usize, const O: usize>(pub Activation);

pub struct LinearLayer<const I: usize, const O: usize> {
    weights: Tensor<Rank2<I, O>>,
    bias: Tensor<Rank1<O>>,
    activation: Activation
}

impl<const I: usize, const O: usize> LayerBuilder for Linear<I, O> {
    type InputShape = Rank1<I>;
    type OutputShape = Rank1<O>;
    type Layer = LinearLayer<I, O>;

    fn build_layer(self, device: &Device) -> Self::Layer {
        let weights = device.sample();
        let bias = device.sample();

        Self::Layer {
            weights,
            bias,
            activation: self.0
        }
    }
}

impl<const I: usize, const O: usize> Layer for LinearLayer<I, O> {
    type InputShape = Rank1<I>;
    type OutputShape = Rank1<O>;

    fn forward(&self, input: Tensor<Rank1<I>>) -> Tensor<Rank1<O>> {
        let a = matmul(input, self.weights.clone()) + self.bias.clone();

        self.activation.apply(a)
    }
    
    fn get_tensors(&self) -> Vec<TensorRef> {
        vec![ self.weights.as_ref(), self.bias.as_ref() ]
    }
}