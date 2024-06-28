use std::sync::Arc;

use crate::{device::Device, tensor::{source::TensorSource, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn sigmoid<S: Shape>(t: Tensor<S>) -> Tensor<S> {
    let device = t.device.clone();

    device.dispatch(TensorSigmoid {
        input: t
    })
}

pub struct TensorSigmoid<S: Shape> {
    pub input: Tensor<S>,
}

impl<S: Shape> TensorOp for TensorSigmoid<S> {
    type OutputShape = S;

    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>) {
        device.back_dispatch(self, output);
    }
}

impl<S: Shape> DispatchTensorOp<TensorSigmoid<S>> for Device {
    fn dispatch(&self, op: TensorSigmoid<S>) -> Tensor<S> {
        let input = op.input.device.get_tensor_buffer(&op.input);
      
        let output = input.iter().map(|i| 1.0 / (1.0 + (-i).exp()) ).collect();

        return op.input.device.clone().allocate_tensor(output, TensorSource::Operation(Arc::new(op)))
    }

    fn back_dispatch(&self, op: &TensorSigmoid<S>, output: &Tensor<S>) {
        let output_gradient = self.get_gradient_buffer(&output);
        let output_buffer = self.get_tensor_buffer(&output);

        let nudge = output_buffer.iter()
                                 .zip(output_gradient)
                                 .map(|(v, d)| *d * v * (1.0 - v))
                                 .collect::<Vec<f32>>();
        
        op.input.device.add_to_gradient(&op.input, &nudge);

        op.input.move_backward();
    }
}