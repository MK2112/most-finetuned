import numpy as np


class NPLinear:
    def __init__(self, i_dim, o_dim, use_bias=True, init="rand"):
        self.i_dim = i_dim
        self.o_dim = o_dim
        self.use_bias = use_bias
        self.init = init
        self.W_grad = 0.0
        self.b_grad = 0.0

        if init == "rand":
            self.W = np.random.random((o_dim, i_dim))
            self.b = np.random.random((o_dim, 1))
        elif init == "zeros":
            self.W = np.zeros((o_dim, i_dim))
            self.b = np.zeros((o_dim, 1))
        elif init == "ones":
            self.W = np.ones((o_dim, i_dim))
            self.b = np.ones((o_dim, 1))

    def forward(self, x):
        return (
            self.W @ x.reshape((-1, 1)) + self.b
            if self.use_bias
            else self.W @ x.reshape((-1, 1))
        )

    def backward(self, x):
        return self.W.T @ x
    
    def flush(self):
        self.W_grad = 0.0
        self.b_grad = 0.0


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self(x) * (1 - self(x))


class Softmax:
    def __call__(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum(axis=0)

    def backward(self, x):
        SM = x.reshape((-1, 1))
        return np.diagflat(x) - np.dot(SM, SM.T)


class Tanh:
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def backward(self, x):
        return 1 - self(x) ** 2
