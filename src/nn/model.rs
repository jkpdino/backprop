use crate::tensor::Tensor;

use super::layers::Layer;

pub struct Model<L: Layer> {
    pub (crate) layer: L
}

impl<L: Layer> Model<L> {
    pub fn forward(&self, input: Tensor<L::InputShape>) -> Tensor<L::OutputShape> {
        self.layer.forward(input)
    }
}