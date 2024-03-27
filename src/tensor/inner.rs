use std::cell::UnsafeCell;

pub struct TensorInner {
    buffer:     UnsafeCell<Vec<f32>>,
    gradient:   UnsafeCell<Vec<f32>>,
}

impl TensorInner {
    pub fn new(buffer: Vec<f32>) -> Self {
        let len = buffer.len();

        Self {
            buffer: UnsafeCell::new(buffer),
            gradient: UnsafeCell::new(vec![0.0; len])
        }
    }

    pub fn buffer(&self) -> &[f32] {
        unsafe { &*self.buffer.get() }
    }

    pub fn buffer_mut(&self) -> &mut [f32] {
        unsafe { &mut *self.buffer.get() }
    }

    pub fn gradient(&self) -> &[f32] {
        unsafe { &*self.gradient.get() }
    }

    pub fn gradient_mut(&self) -> &mut [f32] {
        unsafe { &mut *self.gradient.get() }
    }
}

unsafe impl Sync for TensorInner {}
unsafe impl Send for TensorInner {}