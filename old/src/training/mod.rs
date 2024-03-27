use crate::{inference::Weights, model::Model};

use self::loss::LossFunction;

mod loss;

pub struct Trainer {
    model:      Model,
    weights:    Weights,
    loss:       Box<dyn LossFunction>
}

impl Trainer {
    pub fn run(&mut self, inputs: &[f32], targets: &[f32]) {
        assert_eq!(self.model.input_size(), inputs.len());

        // Count how many activations we have
        // So we can keep track of all of them
        let mut activations =
            self.model.layers().iter()
                      .map(|layer| vec![0.0f32; layer.output_size()])
                      .collect::<Vec<_>>();

        let mut activation_output_idx = 0;
        let mut parameter_idx = 0;
        let mut output_size = 0;

        // Run the forward pass
        for (i, layer) in self.model.layers().iter().enumerate() {
            let input_size = layer.input_size();
            output_size = layer.output_size();
            let parameter_size = layer.parameters_size();

            // Use either the provided inputs or
            // the last outputs
            let input_activations =
                if i == 0 { inputs }
                else { &activations[i - 1]  };

            // Get the output activations
            let output_activations = &mut activations[i];

            let parameters = &self.weights.weights()[parameter_idx..parameter_idx + parameter_size];
            
            layer.forward(parameters, input_activations, output_activations);

            parameter_idx += parameter_size;
            activation_output_idx += output_size;
        }

        // Calculate the loss
        let outputs = &activations[activations.len() - 1];
        let loss = self.loss.forward(outputs, targets);

        // Now we're going to run backprop
        // We have a component of the gradient for each weight
        let mut gradients =
            self.model.layers().iter()
                .map(|layer| vec![0.0f32; layer.parameters_size()])
                .collect::<Vec<_>>();

        let n_layers = gradients.len();

        // Calculate the partial derivative of loss with respect to the outputs
        let mut output_gradient = vec![0.0f32; output_size];
        self.loss.back(outputs, targets, &mut output_gradient);

        for (i, layer) in self.model.layers().iter().enumerate().rev() {
            let parameter_size = layer.parameters_size();
            let parameter_start = parameter_idx - parameter_size;

            let parameters = &self.weights.weights()[parameter_start..parameter_idx];

            // Use either the provided inputs or
            // the last outputs
            let output_activations = &activations[i];

            // Get the gradient
            let gradient = &mut gradients[i];
            layer.back(parameters, output_activations, gradient);

            // Get the last gradient
            let last_gradient = if i == n_layers - 1 {
                output_gradient
            } else {
                gradients[i + 1]
            };

            parameter_idx = parameter_start;
        }
    }
}