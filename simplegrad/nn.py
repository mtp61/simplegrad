from simplegrad.engine import Num, relu, exp
import random


class FeedForward:
    def __init__(self, input_size: int, hidden_size: list[int], output_size: int, activations=None):
        self.layers = [input_size]
        self.layers.extend(hidden_size)
        self.layers.append(output_size)
        if activations is not None:
            self.activations = activations
        else:
            self.activations = len(hidden_size) * ['relu']
        self.initialize_weights()

    def initialize_weights(self):
        def get_rand():
            return random.uniform(-1, 1)

        self.w = [] # weights
        self.b = [] # biases
        for i, (prev_size, size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.w.append([])
            self.b.append([])
            for j in range(size):
                self.b[-1].append(Num(get_rand(), name=f'b_{i}_{j}'))
                self.w[-1].append([])
                for k in range(prev_size):
                    self.w[-1][-1].append(Num(get_rand(), name=f'w_{i}_{j}_{k}'))

    def layer_forward(self, prev: list[Num], layer_w: list[list[Num]], layer_b: list[Num], activation: str) -> list[Num]:
        out = []
        for w, b in zip(layer_w, layer_b):
            z = b
            for prev_node, weight in zip(prev, w):
                z += prev_node * weight
            if activation == 'relu':
                a = relu(z)
            elif activation == 'sigmoid':
                a = Num(1) / (Num(1) + exp(Num(0) - z))
            else:
                a = z
            out.append(a)
        return out

    def forward(self, x: list[Num]) -> list[Num]:
        out = x
        activations = self.activations + ['none']
        for layer_w, layer_b, activation in zip(self.w, self.b, activations):
            out = self.layer_forward(out, layer_w, layer_b, activation)
        return out

    def zero_grad(self):
        for layer in self.w:
            for node in layer:
                for weight in node:
                    weight.zero_grad()
        for layer in self.b:
            for bias in layer:
                bias.zero_grad()

    def update_weights(self, learning_rate: float):
        for layer in self.w:
            for node in layer:
                for weight in node:
                    weight.value -= learning_rate * weight.grad
        for layer in self.b:
            for bias in layer:
                bias.value -= learning_rate * bias.grad


def mse_loss(y_preds: list[Num], ys: list[Num]):
    loss = Num(0)
    for y_pred, y in zip(y_preds, ys):
        error = y - y_pred
        loss += error * error
    return loss / Num(len(ys))

