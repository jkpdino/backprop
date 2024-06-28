use std::sync::Arc;

use crate::tensor::{inner::TensorInner, TensorId};

///
/// A shape-independent reference to a tensor
/// 
pub struct TensorRef {
    pub (crate) id:    TensorId,
    pub (crate) inner: Arc<TensorInner>,
}

impl TensorRef {
    ///
    /// Returns an identifier for the tensor
    /// 
    pub fn id(&self) -> TensorId {
        self.id
    }

    ///
    /// Gets an immutable reference to the tensor buffer
    /// 
    pub fn buffer(&self) -> &[f32] {
        self.inner.buffer()
    }

    ///
    /// Gets a mutable reference to the tensor buffer
    /// 
    pub fn buffer_mut(&self) -> &mut [f32] {
        self.inner.buffer_mut()
    }

    ///
    /// Gets an immutable reference to the tensor gradient buffer
    /// 
    pub fn gradient(&self) -> &[f32] {
        self.inner.gradient()
    }

    ///
    /// Gets a mutable reference to the tensor gradient buffer
    /// 
    pub fn gradient_mut(&self) -> &mut [f32] {
        self.inner.gradient_mut()
    }
}