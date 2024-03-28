use crate::{tensor::{Rank1, Shape, Tensor}, tensor_ops::{cross_entropy_loss, mse}};

pub enum Loss {
    MSE,
    CrossEntropy,
}

impl Loss {
    pub fn apply<S: Shape>(&self, values: Tensor<S>, targets: Tensor<S>) -> Tensor<Rank1<1>> {
        match self {
            Loss::MSE => {
                mse(values, targets)
            }
            Loss::CrossEntropy => {
                cross_entropy_loss(values, targets)
            }
        }
    }
}