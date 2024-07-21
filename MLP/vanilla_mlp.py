import numpy as np
from linear import *
import matplotlib.pyplot as plt


# seed for reproducibility
np.random.seed(42)

# Objective: Given a sine wave, predict the next value of it
x_dim = 5   # Read 5 (consecutive) sine values
y_dim = 10  # Output the next 10 in this series
set_size = 1000

# Create Dataset: set_size sine waves of length x_dim, random amplitude and phase
amplitude = np.random.random((set_size, 1))
phase = np.random.random((set_size, 1))

sines = np.array([amplitude[i] * np.sin(np.arange(x_dim + y_dim) + phase[i]) for i in range(set_size)])  # (set_size, x_dim + y_dim)
        
X = sines[:, :x_dim]  # (set_size, x_dim)
X += np.random.normal(0, 0.05, X.shape) # Add noise to the data
Y = sines[:, x_dim:]  # (set_size, y_dim)

# No data preparation, nothing, just split into train and test
X_train = X[:int(0.8 * set_size)]
Y_train = Y[:int(0.8 * set_size)]
X_test = X[int(0.8 * set_size):]
Y_test = Y[int(0.8 * set_size):]

# Create Model: 2-layer MLP, Sigmoid activation hidden layer, no activation output layer
[layer_1, layer_2, layer_3] = [NPLinear(x_dim, x_dim * y_dim), NPLinear(x_dim * y_dim, x_dim + y_dim), NPLinear(x_dim + y_dim, y_dim)]
sigmoid = Sigmoid()
tanh = Tanh()

# Train Model: 10 epochs, 0.01 learning rate
epochs = 150
lr = 0.01
losses = []

for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i]
        y = Y_train[i]
        
        # Forward Pass
        h_1 = sigmoid(layer_1.forward(x))
        h_2 = tanh(layer_2.forward(h_1.flatten()))
        y_hat = layer_3.forward(h_2.flatten()).flatten()
        
        # MSE as Loss
        loss = np.mean((y - y_hat) ** 2)

        # Backward Pass and Weight Updates
        grad_y_hat = -2 * (y - y_hat) / y_dim
        grad_h_2 = layer_3.backward(grad_y_hat.reshape(-1, 1))
        grad_tanh = tanh.backward(h_2) * grad_h_2
        grad_h_1 = layer_2.backward(grad_tanh)
        grad_sigmoid = sigmoid.backward(h_1) * grad_h_1
        
        layer_3.W -= lr * np.outer(grad_y_hat, h_2.flatten())
        layer_3.b -= lr * grad_y_hat.reshape(-1, 1)

        layer_2.W -= lr * np.outer(grad_tanh.flatten(), h_1.flatten())
        layer_2.b -= lr * grad_tanh

        layer_1.W -= lr * np.outer(grad_sigmoid.flatten(), x.flatten())
        layer_1.b -= lr * grad_sigmoid

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss}')
        losses.append(loss)

# Test Model, plot A single plot: Input in blue, true in green, predicted in red
i = np.random.randint(0, len(X_test))
x = X_test[i]
y = Y_test[i]

h_1 = sigmoid(layer_1.forward(x))
h_2 = tanh(layer_2.forward(h_1))
y_hat = layer_3.forward(h_2).reshape(-1)

plt.plot(np.arange(x_dim), x, 'b')
plt.plot(np.arange(x_dim, x_dim + y_dim), y, 'g')
plt.plot(np.arange(x_dim, x_dim + y_dim), y_hat, 'r')
plt.axhline(0, color='black',linewidth=0.5)
plt.legend(['Input', 'True', 'Predicted'])
plt.show()

plt.plot(losses)
plt.show()

print()
print(f'Input: {x}')
print(f'True: {y}')
print(f'Predicted: {y_hat}')
print(f'Loss: {np.sum((y - y_hat) ** 2)}')


