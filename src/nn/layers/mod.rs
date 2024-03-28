mod linear;
mod combined;
mod conv2d;
mod reshape;

pub use linear::*;
pub use conv2d::*;
pub use reshape::*;

use crate::{device::Device, tensor::{Shape, Tensor}};

pub trait Layer {
    type InputShape: Shape;
    type OutputShape: Shape;

    fn forward(&self, input: Tensor<Self::InputShape>) -> Tensor<Self::OutputShape>;
}

pub trait LayerBuilder {
    type InputShape: Shape;
    type OutputShape: Shape;
    type Layer: Layer<InputShape = Self::InputShape, OutputShape = Self::OutputShape>;

    fn build_layer(self, device: &Device) -> Self::Layer;
}