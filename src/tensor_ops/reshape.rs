use std::{marker::PhantomData, ops::Add, sync::Arc};

use crate::{device::Device, tensor::{source::TensorSource, Rank1, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn reshape<From: Shape, To: Shape>(t: Tensor<From>) -> Tensor<To> {
    assert_eq!(From::SIZE, To::SIZE, "Cannot reshape tensor to a different size");

    let device = t.device.clone();

    device.dispatch(TensorReshape {
        from: t,
        _phantom: PhantomData
    })
}

///
/// Adds two tensors of the same shape
/// 
/// Each element in the output tensor is the sum of the corresponding elements in the input tensors
/// 
#[derive(Clone)]
pub struct TensorReshape<From: Shape, To: Shape> {
    pub from: Tensor<From>,
    _phantom: PhantomData<To>
}

impl<From: Shape, To: Shape> TensorOp for TensorReshape<From, To> {
    type OutputShape = To;

    fn backprop(&self, device: &Device, output: &Tensor<To>) {
        device.back_dispatch(self, output);
    }
}

impl<From: Shape, To: Shape> DispatchTensorOp<TensorReshape<From, To>> for Device {
    fn dispatch(&self, op: TensorReshape<From, To>) -> Tensor<To> {
        let from_buffer = self.get_tensor_buffer(&op.from);

        return self.allocate_tensor(from_buffer.to_owned(), TensorSource::Operation(Arc::new(op)));
    }

    fn back_dispatch(&self, op: &TensorReshape<From, To>, output: &Tensor<To>) {
        let output_gradient = self.get_gradient_buffer(output);
        self.add_to_gradient(&op.from, output_gradient);

        op.from.move_backward();
    }
}