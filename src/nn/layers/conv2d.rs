use crate::{device::Device, nn::Activation, tensor::{Rank1, Rank2, Tensor}, tensor_ops::{conv2d, matmul}};

use super::{Layer, LayerBuilder};

///
/// A convolutional layer.
/// 
/// This layer applies a convolutional transformation to the input tensor, relating each element of the output tensor to a local region of the input tensor.
/// 
/// The layer is parameterized by the kernel shape.
/// 
pub struct Convolution2d<const I1: usize, const I2: usize, const K1: usize, const K2: usize>(pub Activation);

pub struct Convolution2dLayer<const I1: usize, const I2: usize, const K1: usize, const K2: usize> {
    kernel: Tensor<Rank2<K1, K2>>,
    activation: Activation,
}

impl<const I1: usize, const I2: usize, const K1: usize, const K2: usize> LayerBuilder for Convolution2d<I1, I2, K1, K2> {
    type InputShape = Rank2<I1, I2>;
    type OutputShape = Rank2<I1, I2>;
    type Layer = Convolution2dLayer<I1, I2, K1, K2>;

    fn build_layer(self, device: &Device) -> Self::Layer {
        let kernel = device.sample();

        Self::Layer {
            kernel,
            activation: self.0,
        }
    }
}

impl<const I1: usize, const I2: usize, const K1: usize, const K2: usize> Layer for Convolution2dLayer<I1, I2, K1, K2> {
    type InputShape = Rank2<I1, I2>;
    type OutputShape = Rank2<I1, I2>;

    fn forward(&self, input: Tensor<Self::InputShape>) -> Tensor<Self::OutputShape> {
        let a = conv2d(input, self.kernel.clone());

        self.activation.apply(a)
    }
    
    fn get_tensors(&self) -> Vec<crate::tensor::TensorRef> {
        vec![ self.kernel.as_ref() ]
    }
}