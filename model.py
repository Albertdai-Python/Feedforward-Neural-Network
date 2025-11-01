import numpy as np
class Model:
    def __init__(self, structure, eta, batch_size=128):
        # This model uses softmax as its activation function for the last layer and tanh for the other layers
        # Eta: The learning rate which the gradient vector is multiplied by.
        # Structure: List of integers denoting the number of neurons in each layer
        # Batch Size: Number of samples simultaneously trained
        self.structure = structure
        self.eta = eta
        self.batch_size = batch_size
        self.layers = []
        self.layers.append(Layer(structure[0], batch_number=self.batch_size))
        for i in range(len(structure)-1):
            self.layers.append(Layer(structure[i+1], batch_number=self.batch_size, prior=self.layers[i]))

    def sec2h(self, x):
        return 1 / (np.cosh(x)**2)

    def softmax(self, x):
        # x is a numpy array of shape (num_classes, batch_size)
        # Subtract max per column for numerical stability
        shifted_x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=0, keepdims=True)
        return exp_x / sum_exp_x

    def forward_propagate(self, data):
        # For layer L: A^L = tanh(W dot A^(L-1) + B)
        self.layers[0].A = data
        for layer in self.layers[1:-1]:
            layer.Z = layer.W @ layer.prior.A + layer.B
            layer.A = np.tanh(layer.Z)
        layer = self.layers[-1]
        layer.Z = layer.W @ layer.prior.A + layer.B
        layer.A = self.softmax(layer.Z)
        return self.layers[-1].A

    def back_propagate(self, expected):
        weight_gradients = [np.zeros_like(layer.W) for layer in self.layers[1:]]
        bias_gradients = [np.zeros_like(layer.B) for layer in self.layers[1:]]
        delta = (self.layers[-1].A - expected)
        weight_gradients[-1] = delta @ (self.layers[-2].A).T
        self.layers[-1].W -= self.eta * weight_gradients[-1]
        bias_gradients[-1] = np.sum(delta, axis=1, keepdims=True)
        self.layers[-1].B = self.layers[-1].B - self.eta * bias_gradients[-1]
        for idx in range(len(self.layers) - 2, 0, -1):
            delta = (self.layers[idx + 1].W.T @ delta) * self.sec2h(self.layers[idx].Z)
            # delta = (self.layers[idx + 1].W.T @ delta)
            layer_idx = idx - 1
            weight_gradients[layer_idx] = delta @ self.layers[idx].prior.A.T
            self.layers[idx].W = self.layers[idx].W - self.eta * weight_gradients[layer_idx]
            bias_gradients[layer_idx] = np.sum(delta, axis=1, keepdims=True)
            self.layers[idx].B = self.layers[idx].B - self.eta * bias_gradients[layer_idx]

class Layer:
    def __init__(self, number, batch_number, prior=None):
        self.num = number
        self.batch_number = batch_number
        self.Z = np.zeros((self.num, self.batch_number))
        self.A = np.zeros((self.num, self.batch_number))
        if prior:
            self.prior = prior
            self.W = np.random.normal(0, np.sqrt(1 / self.prior.num), (self.num, self.prior.num))
            self.B = np.zeros((self.num, 1))