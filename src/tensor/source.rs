use std::sync::Arc;

use crate::tensor_ops::TensorOp;

use super::Shape;

#[derive(Clone)]
pub enum TensorSource<S: Shape> {
    Constant,
    Operation(Arc<dyn TensorOp<OutputShape = S>>),
}