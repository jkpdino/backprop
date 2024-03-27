use super::LossFunction;

pub struct QuadraticLoss;

impl LossFunction for QuadraticLoss {
    fn forward(&self, activations: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(activations.len(), targets.len());

        let mut loss = 0.0;

        for (activation, target) in activations.iter().zip(targets) {
            let error = activation - target;
            loss += error * error;
        }

        return loss;
    }
    
    fn back(&self, activations: &[f32], targets: &[f32], gradient: &mut [f32]) {
        assert_eq!(activations.len(), targets.len());
        assert_eq!(activations.len(), gradient.len());

        for i in 0..gradient.len() {
            gradient[i] = -2.0 * (activations[i] - targets[i]);
        }
    }
}