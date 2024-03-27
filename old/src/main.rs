use crate::activation::Activation;

mod layers;
mod training;
mod model;
mod inference;
mod activation;

fn main() {
    let input_size = 60;
    let output_size = 20;

    let hidden_size = 120;
    let hidden_depth = 1;

    let mut model = model::Model::default();
    model.add_layer(layers::DenseLayer {
        input_size,
        output_size: hidden_size
    });

    model.add_layer(activation::ReLU.into_layer(hidden_size));

    for _ in 0..hidden_depth {
        model.add_layer(layers::DenseLayer {
            input_size: hidden_size,
            output_size: hidden_size
        });

        model.add_layer(activation::ReLU.into_layer(hidden_size));
    }

    model.add_layer(layers::DenseLayer {
        input_size: hidden_size,
        output_size
    });

    model.add_layer(activation::ReLU.into_layer(output_size));

    println!("Model depth: {} input_size: {} output_size: {} parameters: {}", model.depth(), model.input_size(), model.output_size(), model.parameters_size());

    let inference = model.with_normal_weights();

    let inputs = vec![1.0; input_size];
    let outputs = inference.run(&inputs);

    println!("{outputs:?}");
}
