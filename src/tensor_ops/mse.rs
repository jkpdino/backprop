use std::sync::Arc;

use crate::{device::Device, tensor::{source::TensorSource, Rank1, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn mse<S: Shape>(a: Tensor<S>, targets: Tensor<S>) -> Tensor<Rank1<1>> {
    a.device.clone().dispatch(MeanSquaredError {
        a,
        targets
    })
}

#[derive(Clone)]
pub struct MeanSquaredError<S: Shape> {
    pub a: Tensor<S>,
    pub targets: Tensor<S>,
}

impl<S: Shape> TensorOp for MeanSquaredError<S> {
    type OutputShape = Rank1<1>;
    
    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>) {
        device.back_dispatch(self, output);
    }
}

impl<S: Shape> DispatchTensorOp<MeanSquaredError<S>> for Device {
    fn dispatch(&self, op: MeanSquaredError<S>) -> Tensor<Rank1<1>> {
        let a = self.get_tensor_buffer(&op.a);
        let targets = self.get_tensor_buffer(&op.targets);

        let buffer = a.iter().zip(targets.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (S::SIZE as f32);

        return self.allocate_tensor(vec![buffer], TensorSource::Operation(Arc::new(op)));
    }

    fn back_dispatch(&self, op: &MeanSquaredError<S>, output: &Tensor<Rank1<1>>) {
        // grad Ai = (grad output) * (Ai')
        // output = (1/N) * sum((Ai - Bi)^2)
        // d(output)/d(Ai) = (2/N) * (Ai - Bi)
        // grad Ai = (grad output) * (2/N) * (Ai - Bi)
        // grad Bi = (grad output) * (2/N) * (Bi - Ai)
        let output_gradient = self.get_gradient_buffer(&output);

        let a = self.get_tensor_buffer(&op.a);
        let targets = self.get_tensor_buffer(&op.targets);

        let n = S::SIZE as f32;

        let a_gradient = a.iter()
                          .zip(targets)
                          .map(|(a, target)| output_gradient[0] * (2.0 / n) * (a - target))
                          .collect::<Vec<f32>>();

        self.add_to_gradient(&op.a, &a_gradient);

        let mut target_gradient = a_gradient;

        for target_gradient in target_gradient.iter_mut() {
            *target_gradient *= -1.0;
        }

        self.add_to_gradient(&op.targets, &target_gradient);

        op.a.move_backward();
        op.targets.move_backward();
    }
}