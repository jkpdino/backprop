use super::{Layer, LayerBuilder};

pub struct CombinedLayer<L1: Layer, L2: Layer<InputShape = L1::OutputShape>>
{
    layer1: L1,
    layer2: L2,
}

impl<L1: Layer, L2: Layer<InputShape = L1::OutputShape>> Layer for CombinedLayer<L1, L2> {
    type InputShape = L1::InputShape;

    type OutputShape = L2::OutputShape;

    fn forward(&self, input: crate::tensor::Tensor<Self::InputShape>) -> crate::tensor::Tensor<Self::OutputShape> {
        let intermediate = self.layer1.forward(input);
        return self.layer2.forward(intermediate);
    }
}

impl<L1: LayerBuilder, L2: LayerBuilder<InputShape = L1::OutputShape>> LayerBuilder for (L1, L2) {
    type InputShape = L1::InputShape;
    type OutputShape = L2::OutputShape;
    type Layer = CombinedLayer<L1::Layer, L2::Layer>;

    fn build_layer(self, device: &crate::device::Device) -> Self::Layer {
        return CombinedLayer {
            layer1: self.0.build_layer(device),
            layer2: self.1.build_layer(device),
        }
    }
}

impl<
    L1: LayerBuilder,
    L2: LayerBuilder<InputShape = L1::OutputShape>,
    L3: LayerBuilder<InputShape = L2::OutputShape>
> LayerBuilder for (L1, L2, L3)
{
    type InputShape = L1::InputShape;
    type OutputShape = L3::OutputShape;
    type Layer = CombinedLayer<CombinedLayer<L1::Layer, L2::Layer>, L3::Layer>;

    fn build_layer(self, device: &crate::device::Device) -> Self::Layer {
        return CombinedLayer {
            layer1: CombinedLayer {
                layer1: self.0.build_layer(device),
                layer2: self.1.build_layer(device)
            },
            layer2: self.2.build_layer(device),
        }
    }
}

impl<
    L1: LayerBuilder,
    L2: LayerBuilder<InputShape = L1::OutputShape>,
    L3: LayerBuilder<InputShape = L2::OutputShape>,
    L4: LayerBuilder<InputShape = L3::OutputShape>
> LayerBuilder for (L1, L2, L3, L4)
{
    type InputShape = L1::InputShape;
    type OutputShape = L4::OutputShape;
    type Layer = CombinedLayer<CombinedLayer<CombinedLayer<L1::Layer, L2::Layer>, L3::Layer>, L4::Layer>;

    fn build_layer(self, device: &crate::device::Device) -> Self::Layer {
        return CombinedLayer {
            layer1: CombinedLayer {
                layer1: CombinedLayer {
                    layer1: self.0.build_layer(device),
                    layer2: self.1.build_layer(device),
                },
                layer2: self.2.build_layer(device)
            },
            layer2: self.3.build_layer(device),
        }
    }
}

impl<
    L1: LayerBuilder,
    L2: LayerBuilder<InputShape = L1::OutputShape>,
    L3: LayerBuilder<InputShape = L2::OutputShape>,
    L4: LayerBuilder<InputShape = L3::OutputShape>,
    L5: LayerBuilder<InputShape = L4::OutputShape>,
> LayerBuilder for (L1, L2, L3, L4, L5)
{
    type InputShape = L1::InputShape;
    type OutputShape = L5::OutputShape;
    type Layer = CombinedLayer<CombinedLayer<CombinedLayer<CombinedLayer<L1::Layer, L2::Layer>, L3::Layer>, L4::Layer>, L5::Layer>;

    fn build_layer(self, device: &crate::device::Device) -> Self::Layer {
        return CombinedLayer {
            layer1: CombinedLayer {
                layer1: CombinedLayer {
                    layer1: CombinedLayer {
                        layer1: self.0.build_layer(device),
                        layer2: self.1.build_layer(device),
                    },
                    layer2: self.2.build_layer(device),
                },
                layer2: self.3.build_layer(device)
            },
            layer2: self.4.build_layer(device),
        }
    }
}

impl<
    L1: LayerBuilder,
    L2: LayerBuilder<InputShape = L1::OutputShape>,
    L3: LayerBuilder<InputShape = L2::OutputShape>,
    L4: LayerBuilder<InputShape = L3::OutputShape>,
    L5: LayerBuilder<InputShape = L4::OutputShape>,
    L6: LayerBuilder<InputShape = L5::OutputShape>,
> LayerBuilder for (L1, L2, L3, L4, L5, L6)
{
    type InputShape = L1::InputShape;
    type OutputShape = L6::OutputShape;
    type Layer = CombinedLayer<CombinedLayer<CombinedLayer<CombinedLayer<CombinedLayer<L1::Layer, L2::Layer>, L3::Layer>, L4::Layer>, L5::Layer>, L6::Layer>;

    fn build_layer(self, device: &crate::device::Device) -> Self::Layer {
        return CombinedLayer {
            layer1: CombinedLayer {
                layer1: CombinedLayer {
                    layer1: CombinedLayer {
                        layer1: CombinedLayer {
                            layer1: self.0.build_layer(device),
                            layer2: self.1.build_layer(device),
                        },
                        layer2: self.2.build_layer(device),
                    },
                    layer2: self.3.build_layer(device),
                },
                layer2: self.4.build_layer(device)
            },
            layer2: self.5.build_layer(device),
        }
    }
}