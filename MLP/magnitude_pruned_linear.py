import numpy as np
from linear import *
import matplotlib.pyplot as plt

"""
Weights whose gradients fall below a specific threshold are assumed to have settled relatively close to their final values,
thus they can be pruned based on their magnitudes.
Pruning is carried out iteratively according to a schedule that is one of the model hyperparameters.
Source: https://pml4dc.github.io/iclr2020/pdf/PML4DC2020_17.pdf
"""

# seed for reproducibility
np.random.seed(42)

# Objective: Given a sine wave, predict the next value of it
x_dim = 5   # Read 5 (consecutive) sine values
y_dim = 10  # Output the next 10 in this series
set_size = 1000
train_subset = 0.8

# Params for magnitude-based pruning
prune_beta =  0.01
prune_gamma = 0.001

# Create Dataset: set_size sine waves of length x_dim, random amplitude and phase
amplitude = np.random.random((set_size, 1))
phase = np.random.random((set_size, 1))

sines = np.array([amplitude[i] * np.sin(np.arange(x_dim + y_dim) + phase[i]) for i in range(set_size)])  # (set_size, x_dim + y_dim)
        
X = sines[:, :x_dim]  # (set_size, x_dim)
X += np.random.normal(0, 0.05, X.shape) # Add noise to the data
Y = sines[:, x_dim:]  # (set_size, y_dim)

# No data preparation, nothing, just split into train and test
X_train = X[:int(train_subset * set_size)]
Y_train = Y[:int(train_subset * set_size)]
X_test  = X[int(train_subset * set_size):]
Y_test  = Y[int(train_subset * set_size):]

# Create Model: 2-layer MLP, Sigmoid activation hidden layer, no activation output layer
layers = [NPLinear(x_dim, x_dim * y_dim), NPLinear(x_dim * y_dim, x_dim + y_dim), NPLinear(x_dim + y_dim, y_dim)]
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
        h_1 = sigmoid(layers[0].forward(x))
        h_2 = tanh(layers[1].forward(h_1.flatten()))
        y_hat = layers[2].forward(h_2.flatten()).flatten()
        
        # MSE as Loss
        loss = np.mean((y - y_hat) ** 2)

        # Backward Pass and Weight Updates
        grad_y_hat = -2 * (y - y_hat) / y_dim
        grad_h_2 = layers[2].backward(grad_y_hat.reshape(-1, 1))
        grad_tanh = tanh.backward(h_2) * grad_h_2
        grad_h_1 = layers[1].backward(grad_tanh)
        grad_sigmoid = sigmoid.backward(h_1) * grad_h_1
        
        # This is as easy to read an implementation as I could make it
        # We accumulate gradients now to have pruning itself be more efficient
        layers[2].W_grad += np.outer(grad_y_hat, h_2.flatten())
        layers[2].b_grad += grad_y_hat.reshape(-1, 1)
        layers[1].W_grad += np.outer(grad_tanh.flatten(), h_1.flatten())
        layers[1].b_grad += grad_tanh
        layers[0].W_grad += np.outer(grad_sigmoid.flatten(), x.flatten())
        layers[0].b_grad += grad_sigmoid

        # We go through from 1 to 3. We can do that as we already calculated the gradients
        # in correct (reversed) order above.
        # We just apply that knowledge to the layers now, while pruning them.
        # Prune Update
        for layer in layers:
            # Treating the bias as just another weight
            for attr in ('W', 'b'):
                weight = getattr(layer, attr)
                grad = getattr(layer, f'{attr}_grad')
                mask = (np.abs(weight) < prune_beta) & (np.abs(grad) < prune_gamma)
                weight[mask] = 0  # Zero out only specific elements
                weight[~mask] -= lr * grad[~mask]  # Update non-pruned weights
                setattr(layer, attr, weight)
            layer.flush()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss}')
        losses.append(loss)

# Test Model, plot A single plot: Input in blue, true in green, predicted in red
i = np.random.randint(0, len(X_test))
x = X_test[i]
y = Y_test[i]

h_1 = sigmoid(layers[0].forward(x))
h_2 = tanh(layers[1].forward(h_1))
y_hat = layers[2].forward(h_2).reshape(-1)

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