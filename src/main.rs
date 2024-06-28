use std::error::Error;

use device::Device;
use digit::{DatasetType, MNISTDataset};
use kdam::tqdm;
use nn::Loss;
use simple_moving_average::{SumTreeSMA, SMA};
use tensor::Rank1;

use crate::{nn::optimizer::SgdConfig, tensor::Tensor};

mod digit;

pub mod tensor;
pub mod device;
pub mod tensor_ops;
pub mod nn;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new();

    let model = digit::mnist_model(&device);
    let training_data = MNISTDataset::load(DatasetType::Train).unwrap();
    let test_data = MNISTDataset::load(DatasetType::Test).unwrap();

    let loss_function = Loss::CrossEntropy;

    let epochs = 20;

    let targets = (0..10).map(|n| {
        device.constant(&(0..10).map(|i| if i == n { 1.0 } else { 0.0 }).collect::<Vec<_>>())
    }).collect::<Vec<_>>();

    let mut optimizer = device.build_optimizer(&model, SgdConfig {
        lr: 0.001
    });

    let mut loss_history = vec![];
    let mut training_history = vec![];

    let mut sma = SumTreeSMA::<_, f32, 1000>::new();

    for i in 0..epochs {
        let mut loss = 0.0;

        for (index, row) in tqdm!(training_data.rows().iter().enumerate()) {
            let x_tensor: Tensor<Rank1<784>> = device.constant(&row.pixels);
            let y_tensor = targets[row.label].clone();

            let output = model.forward(x_tensor);

            let loss_value = loss_function.apply(output, y_tensor);

            let loss_num = device.get_tensor_buffer(&loss_value)[0];
            if loss_num.is_nan() {
                println!("loss is nan");
                continue;
            }
            sma.add_sample(loss_num);
            loss += loss_num;

            loss_history.push(sma.get_average());

            optimizer.zero_grad();
            loss_value.back();
            optimizer.step();
        }

        let average_loss = loss / (training_data.rows().len() as f32);
        println!("epoch {i}: loss={average_loss}");

        let mut correct = 0;

        for row in tqdm!(test_data.rows().iter()) {
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

        training_history.push(correct);

        println!("testing: {} / {}", correct, test_data.rows().len());
    }

    let mut writer = csv::Writer::from_path("loss.csv")?;

    for loss in loss_history {
        writer.write_record(&[loss.to_string()])?;
    }

    writer.flush()?;

    let mut writer = csv::Writer::from_path("training.csv")?;

    for correct in training_history {
        writer.write_record(&[correct.to_string()])?;
    }

    writer.flush()?;

    Ok(())
}
