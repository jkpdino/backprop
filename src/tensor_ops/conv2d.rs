use std::sync::Arc;

use crate::{device::Device, tensor::{source::TensorSource, Rank2, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn conv2d<
    const I1: usize,
    const I2: usize,
    const K1: usize,
    const K2: usize
>(
    input: Tensor<Rank2<I1, I2>>,
    kernel: Tensor<Rank2<K1, K2>>
) -> Tensor<Rank2<I1, I2>> {
    input.device.clone().dispatch(TensorConvolve2D {
        input,
        kernel
    })
}

struct TensorConvolve2D<
    const I1: usize,
    const I2: usize,
    const K1: usize,
    const K2: usize>
{
    input: Tensor<Rank2<I1, I2>>,
    kernel: Tensor<Rank2<K1, K2>>,
}

impl<
    const I1: usize,
    const I2: usize,
    const K1: usize,
    const K2: usize
> TensorOp for TensorConvolve2D<I1, I2, K1, K2> {
    type OutputShape = Rank2<I1, I2>;

    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>) {
        device.back_dispatch(self, output)
    }
}

impl<
    const I1: usize,
    const I2: usize,
    const K1: usize,
    const K2: usize
> DispatchTensorOp<TensorConvolve2D<I1, I2, K1, K2>> for Device {
    fn dispatch(&self, op: TensorConvolve2D<I1, I2, K1, K2>) -> Tensor<Rank2<I1, I2>> {
        let input_buffer = self.get_tensor_buffer(&op.input);
        let kernel_buffer = self.get_tensor_buffer(&op.kernel);

        let mut output_buffer = vec![0.0; I1 * I2];

        let x_off = K1 - 1;
        let y_off = K2 - 1;

        for i in 0..(I1 - x_off) {
            for j in 0..(I2 - y_off) {
                for k in 0..K1 {
                    for l in 0..K2 {
                        output_buffer[i * I2 + j] += input_buffer[(i + k) * I2 + (j + l)] * kernel_buffer[k * K2 + l];
                    }
                }
            }
        }
        
        self.allocate_tensor(output_buffer, TensorSource::Operation(Arc::new(op)))
    }

    fn back_dispatch(&self, op: &TensorConvolve2D<I1, I2, K1, K2>, output: &Tensor<Rank2<I1, I2>>) {
        let input_buffer = self.get_tensor_buffer(&op.input);
        let kernel_buffer = self.get_tensor_buffer(&op.kernel);
        
        let output_gradient = self.get_gradient_buffer(output);

        let mut kernel_gradient = vec![0.0; K1 * K2];
        let mut input_gradient = vec![0.0; I1 * I2];

        let x_off = K1 - 1;
        let y_off = K2 - 1;

        // todo: do the math for this
        for i in 0..K1 {
            for j in 0..K2 {
                for k in 0..(I1 - x_off) {
                    for l in 0..(I2 - y_off) {
                        kernel_gradient[i * K2 + j] += input_buffer[(k + i) * I2 + (l + j)] * output_gradient[k * I2 + l];
                        input_gradient[(k + i) * I2 + (l + j)] += kernel_buffer[i * K2 + j] * output_gradient[k * I2 + l];
                    }
                }
            }
        }

        self.add_to_gradient(&op.kernel, &kernel_gradient);
        self.add_to_gradient(&op.input, &input_gradient);
        
        op.input.move_backward();
        op.kernel.move_backward();
    }
}

/*

Convolution Algorithm:
z00 = x00 * k00 + x01 * k01 + x10 * k10 + x11 * k11



*/