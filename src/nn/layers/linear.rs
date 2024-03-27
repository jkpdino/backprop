use crate::{device::Device, tensor::{Rank1, Rank2, Tensor}, tensor_ops::matmul};

pub struct Linear<const I: usize, const O: usize>;

pub struct LinearLayer<const I: usize, const O: usize> {
    weights: Tensor<Rank2<I, O>>,
    bias: Tensor<Rank1<O>>,
}

impl<const I: usize, const O: usize> LinearLayer<I, O> {
    pub fn new(device: &Device) -> Self {
        Self {
            weights: device.sample(),
            bias: device.sample(),
        }
    }
    
    pub fn forward(&self, input: Tensor<Rank1<I>>) -> Tensor<Rank1<O>> {
        let a = matmul(input, self.weights.clone()) + self.bias.clone();

        todo!()
    }
}