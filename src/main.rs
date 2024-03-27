use device::Device;
use rand::thread_rng;
use rand_distr::Distribution;
use tensor::Rank1;
use tensor_ops::relu;

use crate::{tensor::{Rank2, Tensor}, tensor_ops::{matmul, mse, softmax}};

pub mod tensor;
pub mod device;
pub mod tensor_ops;
pub mod nn;

fn main() {
    let device = Device::new();

    // Classify numbers as positive or negative

    let weights_a: Tensor<Rank2<1, 10>> = device.sample();
    let biases_a: Tensor<Rank1<10>> = device.sample();

    let weights_b: Tensor<Rank2<10, 2>> = device.sample();
    let biases_b: Tensor<Rank1<2>> = device.sample();

    let nn = |input: Tensor<Rank1<1>>| {
        let a = relu(matmul(input, weights_a.clone()) + biases_a.clone());
        let b = softmax(matmul(a, weights_b.clone()) + biases_b.clone());

        return b;
    };

    let mut rng = rand::thread_rng();
    let distr = rand_distr::Normal::new(0.0f32, 1.0f32).unwrap();

    let mut losses = vec![];


    for i in 0..100000 {
        let x = distr.sample(&mut rng);
        let y = if x > 0.0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };

        let input = device.constant(vec![x]);
        let target = device.constant(y);

        let output = nn(input);

        let loss = mse(output, target);

        let loss_num = device.get_tensor_buffer(&loss)[0];
    
        device.zero_grad();
        loss.back();
        device.nudge_weights(0.01);

        losses.push(device.get_tensor_buffer(&loss)[0]);
    }

    println!("loss: {}", losses[losses.len() - 10..].iter().sum::<f32>() / 10.0);
}
