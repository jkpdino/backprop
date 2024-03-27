mod quadratic;

pub trait LossFunction {
    fn forward(&self, activations: &[f32], targets: &[f32]) -> f32;
    fn back(&self, activations: &[f32], targets: &[f32], gradient: &mut [f32]);
}