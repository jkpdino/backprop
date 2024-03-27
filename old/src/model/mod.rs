use crate::{inference::{Inference, Weights}, layers::Layer};

/// Represents a model that can be trained and used to make predictions.
/// 
/// The model is made of a sequence of layers, each of which transforms the input
/// activations into output activations.
/// 
/// The model is itself a layer, so it can be used as a sublayer in another model.
pub struct Model {
    layers: Vec<Box<dyn Layer>>
}

impl Model {
    /// Returns the number of layers in the model.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Returns the number of parameters in the model.
    pub fn parameters_size(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameters_size()).sum()
    }

    /// Returns the number of input neurons in the model.
    pub fn input_size(&self) -> usize {
        self.layers.first().map(|layer| layer.input_size()).unwrap_or(0)
    }

    /// Returns the number of output neurons in the model.
    pub fn output_size(&self) -> usize {
        self.layers.last().map(|layer| layer.output_size()).unwrap_or(0)
    }

    /// Adds a new layer to the model.
    pub fn add_layer(&mut self, layer: impl Layer + 'static) {
        self.layers.push(Box::new(layer));
    }

    /// Returns an iterator over the layers of the model.
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }


    /// Creates a new inference object for the model
    /// with the given weights.
    pub fn with_weights(self, weights: Vec<f32>) -> Inference {
        Inference {
            model: self,
            weights: Weights::new(weights)
        }
    }

    /// Creates a new inference object for the model
    /// with random weights.
    pub fn with_normal_weights(self) -> Inference {
        let parameters = self.parameters_size();

        Inference {
            model: self,
            weights: Weights::new_normal(parameters)
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Model {
            layers: Vec::new()
        }
    }
}