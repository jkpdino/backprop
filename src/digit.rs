use std::fs::File;

use crate::{device::Device, nn::{layers::{Convolution2d, Layer, Linear, Reshape}, Activation, Model}, tensor::{Rank1, Rank2}};

pub enum DatasetType {
    Test,
    Train,
}

pub struct MNISTRow {
    pub label: usize,
    pub pixels: Vec<f32>,
}

pub struct MNISTDataset {
    rows: Vec<MNISTRow>
}

impl MNISTDataset {
    pub fn load(ty: DatasetType) -> std::io::Result<MNISTDataset> {
        let url = match ty {
            DatasetType::Test => "data/mnist_test.csv",
            DatasetType::Train => "data/mnist_train.csv",
        };

        let file = File::open(url)?;
        let mut reader = csv::Reader::from_reader(file);

        let mut rows = vec![];

        for record in reader.records() {
            let row = record.unwrap();
            let label = row[0].parse::<usize>().unwrap();

            let pixels = row.iter().skip(1)
                            .map(|p| (p.parse::<u8>().unwrap() as f32) / 255.0)
                            .collect::<Vec<_>>();

            rows.push(MNISTRow {
                label,
                pixels
            })
        }

        Ok(MNISTDataset { rows })
    }

    pub fn rows(&self) -> &[MNISTRow] {
        &self.rows
    }
}

pub fn mnist_model(device: &Device) -> Model<impl Layer<InputShape = Rank1<784>, OutputShape = Rank1<10>>> {
    let model = device.build_model((
        // Reshape the data to a 2d image
        Reshape::<Rank1<784>, Rank2<28, 28>>::new(),

        // Go through a series of convolutions
        (
            Convolution2d::<28, 28, 5, 5>(Activation::ReLU),
            Convolution2d::<28, 28, 7, 7>(Activation::ReLU),
        ),

        // Reshape the data back to a 1d line
        Reshape::<Rank2<28, 28>, Rank1<784>>::new(),

        // Run a series of linear transforms
        (
            Linear::<784, 10>(Activation::Softmax)
        )
    ));

    return model;
}