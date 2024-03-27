use std::{marker::PhantomData, process::Output, sync::Arc};

use crate::{device::Device, tensor_ops::{add::TensorAdd, mse::MeanSquaredError, DispatchTensorOp}};

use self::{inner::TensorInner, source::TensorSource};
pub use self::shape::*;

mod shape;
pub (crate) mod inner;
pub (crate) mod source;

///
/// A unique identifier for a tensor
/// 
/// This is used to link tensors with their gradients
/// 
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TensorId (pub (crate) usize);

///
/// Represents a tensor
/// 
#[derive(Clone)]
pub struct Tensor<S: Shape> {
    pub (crate) id:     TensorId,
    pub (crate) inner:  Arc<TensorInner>,
    pub (crate) device: Device,
    pub (crate) source: TensorSource<S>,

    pub (crate) _shape: PhantomData<S>
}

impl<S: Shape> Tensor<S> {
    ///
    /// Reshapes a tensor into a new shape without changing the data
    /// 
    /// The new shape must have the same number of elements as the original shape
    /// 
    pub fn reshape<Output: Shape>(&self) -> Tensor<Output> {
        assert_eq!(S::SIZE, Output::SIZE);

        todo!()
    }

    pub fn size(&self) -> usize {
        S::SIZE
    }
}

impl<S: Shape> Tensor<S> {
    ///
    /// Runs the backpropagation algorithm on the tensor
    /// 
    /// This will calculate the gradients of the tensor with respect to the output of the operation that created it
    /// 
    pub fn back(&self) {
        // Fill the gradient buffer with ones
        self.device.add_to_gradient(self, &vec![1.0; S::SIZE]);

        self.move_backward();
    }

    pub (crate) fn move_backward(&self) {
        match &self.source {
            TensorSource::Constant => {
                // Constants have no gradients
            }
            TensorSource::Operation(operation) => {
                operation.backprop(&self.device, &self);
            }
        }
    }
}

impl<S: Shape> Drop for Tensor<S> {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 2 {
            self.device.drop_tensor(self.id);
        }
    }
}