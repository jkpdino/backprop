use std::marker::PhantomData;

use crate::tensor::{inner::TensorInner, Shape, TensorRef};

use super::LayerBuilder;

pub struct Reshape<From: Shape, To: Shape> {
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

pub struct ReshapeLayer<From: Shape, To: Shape> {
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

impl<From: Shape, To: Shape> Reshape<From, To> {
    pub fn new() -> Self {
        assert_eq!(From::SIZE, To::SIZE);
        Self {
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}

impl<From: Shape, To: Shape> LayerBuilder for Reshape<From, To> {
    type InputShape = From;
    type OutputShape = To;

    type Layer = ReshapeLayer<From, To>;

    fn build_layer(self, device: &crate::device::Device) -> Self::Layer {
        Self::Layer {
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}

impl<From: Shape, To: Shape> super::Layer for ReshapeLayer<From, To> {
    type InputShape = From;
    type OutputShape = To;

    fn forward(&self, input: crate::tensor::Tensor<Self::InputShape>) -> crate::tensor::Tensor<Self::OutputShape> {
        input.reshape()
    }
    
    fn get_tensors(&self) -> Vec<TensorRef> {
        vec![]
    }
}