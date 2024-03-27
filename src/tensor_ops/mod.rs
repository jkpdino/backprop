pub mod add;
pub mod mse;
pub mod matmul;
mod relu;

use downcast_rs::{impl_downcast, DowncastSync};
pub use mse::mse;
pub use matmul::matmul;
pub use relu::relu;

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