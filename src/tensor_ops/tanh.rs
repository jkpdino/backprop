use std::sync::Arc;

use crate::{device::Device, tensor::{source::TensorSource, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn tanh<S: Shape>(t: Tensor<S>) -> Tensor<S> {
    let device = t.device.clone();

    device.dispatch(TensorTanh {
        input: t
    })
}

pub struct TensorTanh<S: Shape> {
    pub input: Tensor<S>,
}

impl<S: Shape> TensorOp for TensorTanh<S> {
    type OutputShape = S;

    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>) {
        device.back_dispatch(self, output);
    }
}

impl<S: Shape> DispatchTensorOp<TensorTanh<S>> for Device {
    fn dispatch(&self, op: TensorTanh<S>) -> Tensor<S> {
        let input = op.input.device.get_tensor_buffer(&op.input);
      
        let output = input.iter().map(|i| (i.exp() - (-i).exp()) / (i.exp() + (-i).exp()) ).collect();

        return op.input.device.clone().allocate_tensor(output, TensorSource::Operation(Arc::new(op)))
    }

    fn back_dispatch(&self, op: &TensorTanh<S>, output: &Tensor<S>) {
        let output_gradient = self.get_gradient_buffer(&output);
        let output_buffer = self.get_tensor_buffer(&output);

        let nudge = output_buffer.iter()
                                 .zip(output_gradient)
                                 .map(|(v, d)| *d * (1.0 - v*v))
                                 .collect::<Vec<f32>>();
        
        op.input.device.add_to_gradient(&op.input, &nudge);

        op.input.move_backward();
    }
}