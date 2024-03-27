mod dense;

pub use dense::DenseLayer;

type Number = f32;

pub trait Layer {
    fn input_size(&self) -> usize;
    fn parameters_size(&self) -> usize;
    fn output_size(&self) -> usize;

    fn forward(
        &self,
        parameters: &[Number],
        input_activations: &[Number],
        output_activations: &mut [Number]
    );

    fn back(
        &self,
        parameters: &[Number],
        output_activations: &[Number],
        gradient: &mut [Number]
    );
}