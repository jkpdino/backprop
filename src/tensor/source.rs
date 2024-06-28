use std::sync::Arc;

use crate::tensor_ops::TensorOp;

use super::Shape;

pub enum TensorSource<S: Shape> {
    Constant,
    Operation(Arc<dyn TensorOp<OutputShape = S>>),
}

impl<S: Shape> Clone for TensorSource<S> {
    fn clone(&self) -> Self {
        match self {
            Self::Constant => Self::Constant,
            Self::Operation(arg0) => Self::Operation(arg0.clone()),
        }
    }
}