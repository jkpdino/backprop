use crate::activation::Activation;

use super::Layer;

/// A dense layer has n input neurons and m output neurons.
/// It applies a linear transformation to the input activations,
/// using a matrix of weights and a vector of biases.
/// 
/// The dense layer has n*m weights and m biases.
/// 
/// Every output neuron is fully connected to every input neuron.
pub struct DenseLayer {
    pub input_size:  usize,
    pub output_size: usize
}

impl Layer for DenseLayer {
    fn input_size(&self) -> usize {
        self.input_size
    }

    fn parameters_size(&self) -> usize {
        let weights_size = self.input_size * self.output_size;
        let biases_size = self.output_size;

        weights_size + biases_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn forward(
        &self,
        parameters: &[super::Number],
        input_activations: &[super::Number],
        output_activations: &mut [super::Number])
    {
        let weights_size = self.input_size * self.output_size;
        let biases_size = self.output_size;

        let weights = &parameters[0..weights_size];
        let biases = &parameters[weights_size..weights_size + biases_size];

        // For each output neuron
        for j in 0..self.output_size {
            let mut activation = biases[j];

            // For each input neuron
            for i in 0..self.input_size {
                let weight = weights[i * self.output_size + j];
                let input_activation = input_activations[i];

                activation += weight * input_activation;
            }

            output_activations[j] = activation;
        }
    }
    
    fn back(
        &self,
        parameters: &[super::Number],
        input_activations: &[super::Number],
        gradient: &mut [super::Number]
    ) {
        todo!()
    }

    
}