import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer (1 input, 10 hidden nodes)
        self.fc2 = nn.Linear(10, 1)  # Output layer (10 hidden nodes, 1 output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the hidden layer
        x = self.fc2(x)  # Output layer
        return x

# Create the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(net.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Training loop
num_epochs = 1000000
for epoch in range(num_epochs):
    inputs = torch.randn(1, 1)  # Generate a random input value
    target = inputs ** 2  # Calculate the target value (x^2)

    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs, target)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 1000 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the neural network
test_input = torch.tensor([[1.0]])  # Test input
test_output = net(test_input)
print(f'Input: {test_input.item()}, Predicted Output: {test_output.item():.4f}, Expected Output: {test_input.item() ** 2:.4f}')