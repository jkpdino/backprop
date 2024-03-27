use super::Activation;

pub struct ReLU;

impl Activation for ReLU {
    fn forward(&self, x: &[f32], y: &mut [f32]) {
        for i in 0..x.len() {
            y[i] = x[i].max(0.0);
        }
    }
}