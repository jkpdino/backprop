use std::sync::Arc;

use crate::{device::Device, tensor::{source::TensorSource, Rank1, Rank2, Shape, Tensor}};

use super::{DispatchTensorOp, TensorOp};

pub fn matmul<A: Shape, B: Shape>(a: Tensor<A>, b: Tensor<B>) -> Tensor<A::MulOutput>
    where A: MatMul<B>
{
    A::dispatch(a, b)
}

pub trait MatMul<S: Shape>: Shape + Sized {
    type MulOutput: Shape;

    fn dispatch(a: Tensor<Self>, b: Tensor<S>) -> Tensor<Self::MulOutput>;
}

impl<const A: usize, const B: usize, const C: usize> MatMul<Rank2<B, C>> for Rank2<A, B> {
    type MulOutput = Rank2<A, C>;

    fn dispatch(_a: Tensor<Self>, _b: Tensor<Rank2<B, C>>) -> Tensor<Self::MulOutput> {
        todo!()
    }
}

impl<const A: usize, const B: usize> MatMul<Rank2<A, B>> for Rank1<A> {
    type MulOutput = Rank1<B>;

    fn dispatch(a: Tensor<Self>, b: Tensor<Rank2<A, B>>) -> Tensor<Self::MulOutput> {
        a.device.clone().dispatch(TensorMatMul {
            lhs: a,
            rhs: b
        })
    }
}

pub struct TensorMatMul<A: Shape, B: Shape> where A: MatMul<B> {
    pub lhs: Tensor<A>,
    pub rhs: Tensor<B>
}

impl<const A: usize, const B: usize> TensorOp for TensorMatMul<Rank1<A>, Rank2<A, B>> {
    type OutputShape = Rank1<B>;

    fn backprop(&self, device: &Device, output: &Tensor<Self::OutputShape>) {
        device.back_dispatch(self, output);
    }
}

impl<const A: usize, const B: usize> DispatchTensorOp<TensorMatMul<Rank1<A>, Rank2<A, B>>> for Device  {
    fn dispatch(&self, op: TensorMatMul<Rank1<A>, Rank2<A, B>>) -> Tensor<Rank1<B>> {
        let lhs = self.get_tensor_buffer(&op.lhs);
        let rhs = self.get_tensor_buffer(&op.rhs);

        let mut buffer = vec![0.0; B];

        for i in 0..B {
            for j in 0..A {
                buffer[i] += lhs[j] * rhs[(j * B) + i];
            }
        }

        return op.lhs.device.clone().allocate_tensor(buffer, TensorSource::Operation(Arc::new(op)));
    }

    fn back_dispatch(&self, op: &TensorMatMul<Rank1<A>, Rank2<A, B>>, output: &Tensor<Rank1<B>>) {
        let output_gradient = self.get_gradient_buffer(&output);

        let lhs = self.get_tensor_buffer(&op.lhs);
        let rhs = self.get_tensor_buffer(&op.rhs);

        let mut activation_gradient = vec![0.0f32; lhs.len()];

        //println!("{output_gradient:?}");

        //
        //       b00 b01
        //       b10 b11
        // a0 a1 a0b00 + a1b10  a0b01 + a1b11
        //
        for i in 0..A {
            for j in 0..B {
                let del_output = output_gradient[j];
                let weight = rhs[(i * B) + j];

                activation_gradient[i] += del_output * weight;
            }
        }

        //println!("activation: {activation_gradient:?}");

        let mut weights_gradient = vec![0.0f32; rhs.len()];
        for i in 0..A {
            for j in 0..B {
                let del_output = output_gradient[j];

                weights_gradient[(i * B) + j] = del_output * lhs[i];
            }
        }

        //println!("weights: {weights_gradient:?}");

        self.add_to_gradient(&op.lhs, &activation_gradient);
        self.add_to_gradient(&op.rhs, &weights_gradient);

        op.lhs.move_backward();
        op.rhs.move_backward();
    }
}