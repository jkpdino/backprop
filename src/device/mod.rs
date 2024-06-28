use std::{cell::UnsafeCell, collections::HashMap, sync::Arc};

use rand_distr::Distribution;

use crate::{nn::{layers::{Layer, LayerBuilder}, optimizer::OptimizerConfig, Model}, tensor::{inner::TensorInner, source::TensorSource, Shape, Tensor, TensorId}};

pub struct DeviceInner {
    tensor_buffers: HashMap<TensorId, Arc<TensorInner>>,
    tensor_allocated: usize,
}

#[derive(Clone)]
pub struct Device {
    inner: Arc<UnsafeCell<DeviceInner>>,
}

impl Device {
    pub fn get_tensor_buffer<S: Shape>(&self, tensor: &Tensor<S>) -> &[f32] {
        self.inner().tensor_buffers.get(&tensor.id).unwrap().buffer()
    }

    pub fn get_gradient_buffer<S: Shape>(&self, tensor: &Tensor<S>) -> &[f32] {
        self.inner().tensor_buffers.get(&tensor.id).unwrap().gradient()
    }

    pub (crate) fn add_to_gradient<S: Shape>(&self, tensor: &Tensor<S>, buffer: &[f32]) {
        let gradient = self.inner().tensor_buffers.get(&tensor.id).unwrap().gradient_mut();

        assert_eq!(gradient.len(), buffer.len());

        for (g, b) in gradient.iter_mut().zip(buffer.iter()) {
            *g += b;
        }
    }
}

impl Device {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(UnsafeCell::new(DeviceInner {
                tensor_buffers: HashMap::new(),
                tensor_allocated: 0
            }))
        }
    }

    ///
    /// Builds a model from a LayerBuilder, initializing all the weights.
    /// 
    pub fn build_model<L: LayerBuilder>(&self, builder: L) -> Model<L::Layer> {
        let layer = builder.build_layer(self);

        Model { layer }
    }

    /// 
    /// Builds an optimizer from a model and an OptimizerConfig
    /// 
    pub fn build_optimizer<O: OptimizerConfig>(&self, model: &Model<impl Layer>, cfg: O) -> O::Optimizer {
        let tensors = model.layer.get_tensors();

        cfg.build_optimizer(tensors, self.clone())
    }

    /// 
    /// Resets all gradient vectors
    /// 
    pub fn zero_grad(&self) {
        let inner = self.inner();

        for buffer in inner.tensor_buffers.values() {
            buffer.gradient_mut().fill(0.0);
        }
    }
}

// Tensor constructors
impl Device {
    ///
    /// Create a tensor with the given data
    /// 
    pub fn constant<S: Shape>(&self, data: &[f32]) -> Tensor<S> {
        self.allocate_tensor(data.to_vec(), TensorSource::Constant)
    }

    ///
    /// Create a tensor with random normal data
    /// 
    pub fn sample<S: Shape>(&self) -> Tensor<S> {
        let mut rng = rand::thread_rng();
        let distr = rand_distr::Normal::new(0.0f32, 0.01).unwrap();

        let data = (0..S::SIZE).map(|_| distr.sample(&mut rng)).collect();

        self.allocate_tensor(data, TensorSource::Constant)
    }
}

impl Device {
    pub fn allocate_tensor<S: Shape>(&self, data: Vec<f32>, source: TensorSource<S>) -> Tensor<S> {
        assert_eq!(S::SIZE, data.len());

        let inner = self.inner();

        let tensor_id = TensorId(inner.tensor_allocated);
        inner.tensor_allocated += 1;

        let tensor_inner = Arc::new(TensorInner::new(data));

        inner.tensor_buffers.insert(tensor_id, tensor_inner.clone());

        Tensor {
            id: tensor_id,
            inner: tensor_inner,
            device: self.clone(),
            source,
            _shape: std::marker::PhantomData,
            
        }
    }

    pub (crate) fn drop_tensor(&self, id: TensorId) {
        self.inner().tensor_buffers.remove(&id);
    }
}

impl Device {
    fn inner(&self) -> &mut DeviceInner {
        unsafe { &mut *self.inner.get() }
    }
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}