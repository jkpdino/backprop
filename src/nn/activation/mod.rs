use crate::{tensor::{Shape, Tensor}, tensor_ops::{relu, sigmoid, softmax, tanh}};

#[derive(Debug)]
pub enum Activation {
    ///
    /// Rectified Linear Unit activation. Returns a zero if the input is less than zero, otherwise returns the input.
    /// 
    /// ReLU is the most commonly used activation function, as it approximates biological neurons.
    /// 
    ReLU,

    ///
    /// Softmax activation. Returns a vector of probabilities that sum to one.
    /// 
    /// Most often used as the output layer of a neural network, as it can be interpreted as a probability distribution.
    /// 
    Softmax,

    ///
    /// Sigmoid activation. Returns a value between zero and one.
    /// 
    /// This activation function is not often used, as it:
    /// 1) Flattens around the origin, leading to a slowdown in learning
    /// 2) Outputs are not zero-centered, leading to zig-zagging in the gradient descent
    /// 
    /// It can still be used for probability in a binary classification problem.
    /// 
    Sigmoid,

    ///
    /// Hyperbolic Tangent activation. Returns a value between -1 and 1.
    /// 
    /// This activation function is often used in RNNs, as it is zero-centered and outputs are in a range that is more likely to be useful.
    /// However, it still has the vanishing gradient problem.
    /// 
    Tanh,
}

impl Activation {
    pub fn apply<S: Shape>(&self, input: Tensor<S>) -> Tensor<S> {
        match self {
            Activation::ReLU => relu(input),
            Activation::Softmax => softmax(input),
            Activation::Sigmoid => sigmoid(input),
            Activation::Tanh => tanh(input),
        }
    }
}