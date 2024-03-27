use super::Activation;

pub struct Softmax;

impl Activation for Softmax {
    fn forward(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), y.len());

        for i in 0..x.len() {
            y[i] = x[i].exp();
        }

        let sum: f32 = y.iter().sum();

        println!("{y:?} {sum}");

        for i in 0..y.len() {
            y[i] /= sum;
        }
    }
}