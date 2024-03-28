pub mod add;
pub mod mse;
pub mod matmul;
mod relu;
mod softmax;
mod sigmoid;
mod tanh;
mod cross_entropy;
mod conv2d;
//mod pool;

use downcast_rs::{impl_downcast, DowncastSync};
pub use mse::mse;
pub use cross_entropy::cross_entropy_loss;
pub use matmul::matmul;
pub use relu::relu;
pub use softmax::softmax;
pub use sigmoid::sigmoid;
pub use tanh::tanh;
pub use conv2d::conv2d;
//pub use pool::{maxpool, maxpool2d};

use crate::{device::Device, tensor::{Shape, Tensor}};

pub trait TensorOp: DowncastSync {
    type OutputShape: Shape;

    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>);
}

impl_downcast!(sync TensorOp assoc OutputShape);

pub trait DispatchTensorOp<T: TensorOp> {
    fn dispatch(&self, op: T) -> Tensor<T::OutputShape>;
    fn back_dispatch(&self, op: &T, output: &Tensor<T::OutputShape>);
}