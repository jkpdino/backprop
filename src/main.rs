use device::Device;
use digit::{DatasetType, MNISTDataset};
use kdam::tqdm;
use nn::Loss;
use rand_distr::Distribution;
use tensor::Rank1;

use crate::{nn::{layers::{Convolution2d, Linear, Reshape}, Activation}, tensor::{Rank2, Tensor}};

mod digit;

pub mod tensor;
pub mod device;
pub mod tensor_ops;
pub mod nn;

fn main() {
    let device = Device::new();

    let model = device.build_model((
        Reshape::<Rank1<784>, Rank2<28, 28>>::new(),
        Convolution2d::<28, 28, 3, 3>,
        Convolution2d::<28, 28, 3, 3>,
        Reshape::<Rank2<28, 28>, Rank1<784>>::new(),
        Linear::<784, 10>(Activation::Softmax),
        //Linear::<49, 10>(Activation::Softmax)
    ));

    let loss_function = Loss::CrossEntropy;

    let epochs = 5;

    let mnist = MNISTDataset::load(DatasetType::Train).unwrap();

    let targets = (0..10).map(|n| {
        let mut target = vec![0.0f32; 10];
        target[n] = 1.0f32;
        device.constant(&target)
    }).collect::<Vec<_>>();

    for i in 0..epochs {
        let mut loss = 0.0;

        for row in tqdm!(mnist.rows().iter()) {
            let x_tensor: Tensor<Rank1<784>> = device.constant(&row.pixels);
            let y_tensor = targets[row.label].clone();

            let output = model.forward(x_tensor);

            let loss_value = loss_function.apply(output, y_tensor);

            let loss_num = device.get_tensor_buffer(&loss_value)[0];
            if loss_num.is_nan() {
                println!("loss is nan");
                break;
            }
            loss += loss_num;

            device.zero_grad();
            loss_value.back();
            device.nudge_weights(0.01);
        }

        let average_loss = loss / (mnist.rows().len() as f32);
        println!("epoch {i}: loss={average_loss}");
    }

    let mut correct = 0;

    let test = MNISTDataset::load(DatasetType::Test).unwrap();

    for row in tqdm!(test.rows().iter()) {
        let x_tensor: Tensor<Rank1<784>> = device.constant(&row.pixels);

        let output = model.forward(x_tensor);
        let output_buffer = device.get_tensor_buffer(&output);

        let mut max = 0.0;
        let mut max_index = 0;

        //println!("output: {:?} {}", output_buffer, row.label);

        for (i, n) in output_buffer.iter().enumerate() {
            if n > &max {
                max = *n;
                max_index = i;
            }
        }

        if max_index == row.label {
            correct += 1;
        }
    }

    println!("testing: {} / {}", correct, test.rows().len());
}
