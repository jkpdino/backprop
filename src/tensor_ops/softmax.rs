use std::sync::Arc;

use crate::{device::Device, tensor::{source::TensorSource, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn softmax<S: Shape>(t: Tensor<S>) -> Tensor<S> {
    let device = t.device.clone();

    device.dispatch(TensorSoftmax {
        input: t
    })
}

pub struct TensorSoftmax<S: Shape> {
    pub input: Tensor<S>,
}

impl<S: Shape> TensorOp for TensorSoftmax<S> {
    type OutputShape = S;

    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>) {
        device.back_dispatch(self, output);
    }
}

impl<S: Shape> DispatchTensorOp<TensorSoftmax<S>> for Device {
    fn dispatch(&self, op: TensorSoftmax<S>) -> Tensor<S> {
        let input = op.input.device.get_tensor_buffer(&op.input);

        let factor = input.iter().cloned().reduce(f32::max).unwrap();
        let mut output = input.iter().map(|i| (i - factor).exp() ).collect::<Vec<_>>();
        let sum = output.iter().sum::<f32>();

        for i in output.iter_mut() {
            *i /= sum;
        }

        return op.input.device.clone().allocate_tensor(output, TensorSource::Operation(Arc::new(op)))
    }

    fn back_dispatch(&self, op: &TensorSoftmax<S>, output: &Tensor<S>) {
        /* How to do the gradient of a softmax? */
        /* Remember, Xi = (e^Ai) / (sum e^Aj) = u / v  */
        /* del Xi / del Ak = (u'v - uv')/(v^2)   */
        // v' = 1
        // i == k: u' = e^Ak
        //
        // Xi' = (e^Ak * sum e^Aj - e^Ak * e^Ak) / (sum e^Aj)^2
        //     = (e^Ak) / (sum e^Aj) * (sum e^Aj - e^Ak) / (sum e^Aj)
        //     = softmax(e^Ak) * (1 - softmax(e^Ak))
        //
        // i != k: u' = 0
        //
        // Xi' = (0 - e^Ai * e^Ak) / (sum e^Aj)^2
        //     = -(e^Ai / sum e^Aj) * (e^Ak / sum e^Aj)
        //     = -softmax(Ai) * softmax(Ak)

        let output_gradient = self.get_gradient_buffer(&output);
        let output_buffer = self.get_tensor_buffer(&output);

        // grad Ai = sum[j] (grad Xj * del Xj / del Ai)

        let mut gradient = vec![0.0f32; op.input.size()];

        for j in 0..output_buffer.len() {
            let softmax_j = output_buffer[j];

            for i in 0..gradient.len() {
                let softmax_i = output_buffer[i];
                let factor = if i == j { 1.0 } else { 0.0 };

                gradient[i] = output_gradient[j] * ( (factor - softmax_i) * softmax_j );
            }
            
            op.input.device.add_to_gradient(&op.input, &gradient);
        }

        op.input.move_backward();
    }
}