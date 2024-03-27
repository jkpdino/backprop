# How a neural network works

A neural network takes $n$ inputs, and runs a series of mathematical functions based on parameters on them to produce $m$ outputs. For each set of inputs given, there is an expected set of outputs. During training, an error term is calculated based on the difference between the outputs and expected outputs. Using the process of recursive descent, the algorithm changes the weights to minimize the error term.

## Gradient Descent

Gradient descent runs the neural network over and over again, changing the weights to minimize the error term. A function called the loss function is calculated in terms of all the parameters and inputs to the neural network. The loss function measures how far the outputs of the neural network are from the expected outputs. We will call this loss function $L$. During gradient descent, the goal is to minimize this loss function.

Imagine trying to minimize a function $f(x)$. We could start by making a guess $x_0$ for the value that minimizes the function. Then, we could calculate the slope of the function at that point $\frac{df(x)}{dx}$. This slope will point upwards, so if we move in the opposite direction by some amount $\delta$, we will get closer to the minimum. Then we can take $x_1=x_0-\delta\frac{df(x_0)}{dx}$, and similarly take $x_2, x_3, x_4$. The more we repeat this process, the closer we will get to the minimum.

Now imagine we have a function $f(x, y, z)$. We can use the same exact process by starting with the guesses $(x_0, y_0, z_0)$. We can nudge each of the guesses in the opposite direction of the partial derivative to minimize the function. By doing this, we get $x_1=x_1-\delta\frac{\partial f}{\partial x}$, $y_1=y_1-\delta\frac{\partial f}{\partial y}$, and $z_1=z_1-\delta\frac{\partial f}{\partial z}$. Now, lets define vectors $\vec{x_0}=[x_0,y_0,z_0]$ and $\vec{x_1}=[x_1,y_1,z_1]$. Then, $\vec{x_1}=\vec{x_0}-\delta \cdot [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}]$. This vector of partial derivatives is what is known as the *gradient* of the function, and can be denoted as $\nabla f(\vec{x})$, giving us the final equation which works for any number of parameters: $$\vec{x}_{n+1}=\vec{x}_{n}-\delta \cdot \nabla f(\vec{x}_n)$$

## Training

## Loss Function

Now that we understand what gradient descent is, we must try to understand the function that we are minimizing. The first step in training a neural network is to calculate the loss function. The loss function is a function of all the outputs of the neural network, and the targets for each of the outputs. We will use the notations:
$$\text{Outputs of the neural network}: z^{(N)}_i$$
$$\text{Targets for the neural network}: t^{(N)}_i$$
$$\text{Loss function}: L(z^{(N)}_i | t^{(N)}_i)$$

There are a few functions used for the loss function:

### Mean-Squared Error

Mean squared error sums up all the squares of the differences