use std::fs::File;

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

pub fn mnist_model() {
    /*
    let model = device.build_model((
        (Conv2d::<3, 3>, MaxPool2d::<2, 2>),
        (Conv2d::<3, 3>, MaxPool2d::<2, 2>),
        (Conv2d::<3, 3>, MaxPool2d::<2, 2>),
        Reshape::<Rank2<7, 7>, Rank1<49>>,
        Linear::<49, 196>(Activation::ReLU),
        Linear::<196, 10>(Activation::Softmax)
    ));
     */
}