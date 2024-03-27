use crate::model::Model;

mod weights;

pub use weights::Weights;

pub struct Inference {
    pub(crate) model: Model,
    pub(crate) weights: Weights
}

impl Inference {
    pub fn run(&self, inputs: &[f32]) -> Vec<f32> {
        assert_eq!(self.model.input_size(), inputs.len());

        let mut activations = inputs.to_vec();

        let mut parameter_index = 0;

        for layer in self.model.layers() {
            let input_activations = &activations;
            let output_size = layer.output_size();
            let mut output_activations = vec![0.0; output_size];

            let parameters = &self.weights.weights()[parameter_index..parameter_index + layer.parameters_size()];

            layer.forward(parameters, input_activations, &mut output_activations);

            activations = output_activations;
            parameter_index += layer.parameters_size();
        }

        return activations;
    }
}