from numpy import random

from src.value import Value


class Neuron:
    def __init__(self, num_inputs: int, layer_num: int, neuron_num: int):
        self.w = [Value(random.uniform(-1.0, 1.0), label=f"l{layer_num}n{neuron_num}w{i}") for i in range(num_inputs)]
        self.b = Value(random.uniform(-1.0, 1.0), label=f"l{layer_num}n{neuron_num}b")

    def __call__(self, x: list[float | int | Value], example_num: int) -> Value:
        assert len(x) == len(self.w), f"Input size must be equal to weight size x.size = {len(x)}, w.size = {len(self.w)}"
        act = sum((self.w[i] * (x[i] if isinstance(x[i], Value) else Value(x[i], label=f"e{example_num}x{i}")) for i in range(len(x))), self.b) 
        out = act.tanh()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]
    
class Layer:
    def __init__(self, num_inputs: int, num_neurons: int, layer_num: int):
        self.neurons = [Neuron(num_inputs, layer_num, i) for i in range(num_neurons)]
    
    def __call__(self, x: list[float | int | Value], example_num: int) -> list[Value]:
        outs = [n(x, example_num) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, num_inputs: int, num_outputs: list[int]):
        sz = [num_inputs] + num_outputs
        self.layers = [Layer(sz[i], sz[i+1], i) for i in range(len(num_outputs))]

    def __call__(self, x: list[float | int | Value], example_num: int) -> list[Value]:
        for layer in self.layers:
            x = layer(x, example_num)
        return x
    
    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
