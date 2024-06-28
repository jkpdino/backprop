use std::sync::Arc;

use crate::{device::Device, tensor::{source::TensorSource, Rank1, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn cross_entropy_loss<S: Shape>(a: Tensor<S>, targets: Tensor<S>) -> Tensor<Rank1<1>> {
    a.device.clone().dispatch(TensorCrossEntropyLoss {
        a,
        targets
    })
}

#[derive(Clone)]
pub struct TensorCrossEntropyLoss<S: Shape> {
    pub a: Tensor<S>,
    pub targets: Tensor<S>,
}

impl<S: Shape> TensorOp for TensorCrossEntropyLoss<S> {
    type OutputShape = Rank1<1>;
    
    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>) {
        device.back_dispatch(self, output);
    }
}

impl<S: Shape> DispatchTensorOp<TensorCrossEntropyLoss<S>> for Device {
    fn dispatch(&self, op: TensorCrossEntropyLoss<S>) -> Tensor<Rank1<1>> {
        let a = self.get_tensor_buffer(&op.a);
        let targets = self.get_tensor_buffer(&op.targets);

        //println!("a: {:?}", a);

        let buffer = -a.iter().zip(targets.iter()).map(|(a, t)| t * a.ln()).sum::<f32>();

        return self.allocate_tensor(vec![buffer], TensorSource::Operation(Arc::new(op)));
    }

    fn back_dispatch(&self, op: &TensorCrossEntropyLoss<S>, output: &Tensor<Rank1<1>>) {
        /*
            z = -sum (target * ln(a))
            dz/da = -sum (target / a)
         */
        let output_gradient = self.get_gradient_buffer(&output);

        let a = self.get_tensor_buffer(&op.a);
        let targets = self.get_tensor_buffer(&op.targets);

        let a_gradient = a.iter()
                          .zip(targets)
                          .map(|(a, target)| if a == &0.0 { 0.0 } else { -output_gradient[0] * target / a })
                          .collect::<Vec<f32>>();

        self.add_to_gradient(&op.a, &a_gradient);

        op.a.move_backward();
    }
}