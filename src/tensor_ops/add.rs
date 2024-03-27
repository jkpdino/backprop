use std::{ops::Add, sync::Arc};

use crate::{device::Device, tensor::{source::TensorSource, Rank1, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

///
/// Adds two tensors of the same shape
/// 
/// Each element in the output tensor is the sum of the corresponding elements in the input tensors
/// 
#[derive(Clone)]
pub struct TensorAdd<S: Shape> {
    pub lhs: Tensor<S>,
    pub rhs: Tensor<S>,
}

impl<S: Shape> TensorOp for TensorAdd<S> {
    type OutputShape = S;

    fn backprop(&self, device: &Device, output: &Tensor<S>) {
        device.back_dispatch(self, output);
    }
}

impl<const A: usize> Add for Tensor<Rank1<A>> {
    type Output = Tensor<Rank1<A>>;

    fn add(self, rhs: Self) -> Self::Output {
        self.device.dispatch(TensorAdd {
            lhs: self.clone(),
            rhs: rhs.clone(),
        })
    }
}

impl<S: Shape> DispatchTensorOp<TensorAdd<S>> for Device {
    fn dispatch(&self, op: TensorAdd<S>) -> Tensor<S> {
        let lhs_buffer = self.get_tensor_buffer(&op.lhs);
        let rhs_buffer = self.get_tensor_buffer(&op.rhs);

        let buffer = lhs_buffer.iter().zip(rhs_buffer.iter()).map(|(a, b)| a + b).collect();

        return self.allocate_tensor(buffer, TensorSource::Operation(Arc::new(op)));
    }

    fn back_dispatch(&self, op: &TensorAdd<S>, output: &Tensor<S>) {
        let output_gradient = self.get_gradient_buffer(&output);

        // grad Ai = grad C
        // grad Bi = grad C
        self.add_to_gradient(&op.lhs, output_gradient);
        self.add_to_gradient(&op.rhs, output_gradient);

        op.lhs.move_backward();
        op.rhs.move_backward();
    }
}