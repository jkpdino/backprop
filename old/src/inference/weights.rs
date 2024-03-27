use rand_distr::{Distribution, Normal};

pub struct Weights {
    weights: Vec<f32>
}

impl Weights {
    /// Create a new set of weights with the given values.
    pub fn new(weights: Vec<f32>) -> Weights {
        Weights {
            weights
        }
    }

    /// Create a new set of weights with random values from a normal distribution.
    pub fn new_normal(n_weights: usize) -> Weights {
        let normal = Normal::new(0.0f32, 1.0f32).unwrap();
        let mut rng = rand::thread_rng();

        let weights =
            normal.sample_iter(&mut rng)
                .take(n_weights)
                .collect();

        Weights {
            weights
        }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}