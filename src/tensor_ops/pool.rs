use std::sync::Arc;

use crate::{device::Device, tensor::{source::TensorSource, Rank2, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn maxpool<S: Shape>(tensor: Tensor<S>, kernel_size: f32) {


}

pub fn maxpool2d() {
    
}


pub struct TensorMaxpool<S: Shape> {
    pub input: Tensor<S>,
    pub kernel_size: f32,
}

impl<const I1: usize, const I2: usize> TensorOp for TensorMaxpool<Rank2<I1, I2>> {
    type OutputShape = Rank2<I1, I2 + 1>;
    
    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>) {
        device.back_dispatch(self, output)
    }
}

impl<S: Shape> DispatchTensorOp<TensorMaxpool<S>> for Device {
    fn dispatch(&self, op: TensorMaxpool<S>) -> Tensor<S> {
        let input = op.input.device.get_tensor_buffer(&op.input);

        let last_dimension = S::last_dim();

        let base = 0;

        while base < input.len() {
            let len = last_dimension.min(input.len() - base);

            let bucket_base = base;

            while bucket_base < base + len {
                let bucket_len = last_dimension.min(input.len() - bucket_base);

                let mut max = f32::MIN;

                for i in 0..bucket_len {
                    max = max.max(input[bucket_base + i]);
                }

                for i in 0..bucket_len {
                    input[bucket_base + i] = max;
                }

                bucket_base += last_dimension;
            
            }
        }

        return op.input.device.clone().allocate_tensor(output, TensorSource::Operation(Arc::new(op)))
    }

    fn back_dispatch(&self, op: &TensorMaxpool<S>, output: &Tensor<S>) {
        let input = op.input.device.get_tensor_buffer(&op.input);

        let last_dimension = S::last_dim();

        let base = 0;

        while base < input.len() {
            let len = last_dimension.min(input.len() - base);

            let bucket_base = base;

            while bucket_base < base + len {
                let bucket_len = last_dimension.min(input.len() - bucket_base);

                let mut max = f32::MIN;

                for i in 0..bucket_len {
                    max = max.max(input[bucket_base + i]);
                }

                for i in 0..bucket_len {
                    input[bucket_base + i] = max;
                }

                bucket_base += last_dimension;
            
            }
        }

        op.input.move_backward();
    }
}