import math
import random
from src.nanograd.value import Value


class Neuron:
    def __init__(self, num_inputs: int, layer_idx: int, neuron_idx: int, **kwargs):
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx

        weights_init = None
        match kwargs.get("initilization", "normal"):
            case "uniform":
                weights_init = lambda: random.uniform(-1.0, 1.0)
            case "normal":
                mean = kwargs.get("mean", 0.0)
                std = kwargs.get("std", 1.0)
                weights_init = lambda: random.normalvariate(mean, std)
            case "constant":
                weights_init = lambda: kwargs.get("value", 0.0)
            case "xavier_uniform":
                gain = kwargs.get("gain", 1.0)
                weights_init = lambda: random.uniform(-gain, gain) * math.sqrt(
                    (6.0 / (num_inputs + 1))
                )
            case "xavier_normal":
                gain = kwargs.get("gain", 1.0)
                weights_init = lambda: random.normalvariate(0.0, gain) * math.sqrt(
                    (2.0 / (num_inputs + 1))
                )
            case "truncate_normal":
                mean = kwargs.get("mean", 0.0)
                std = kwargs.get("std", 1.0)
                min_val = kwargs.get("min_val", -2.0)
                max_val = kwargs.get("max_val", 2.0)
                weights_init = lambda: min(
                    max(random.normalvariate(mean, std), min_val), max_val
                )
            case _:
                raise ValueError(
                    f"Unknown initilization: {kwargs.get('initilization', 'uniform')}"
                )
        self.w = [
            Value(weights_init(), label=f"l{layer_idx}n{neuron_idx}w{input_idx}")
            for input_idx in range(num_inputs)
        ]
        self.b = Value(weights_init(), label=f"l{layer_idx}n{neuron_idx}b")

    def __call__(
        self,
        x: list[Value],
        example_idx: int,
        activation_fn: str = "linear",
    ) -> Value:
        assert len(x) == len(
            self.w
        ), f"Input size must be equal to weight size x.size = {len(x)}, w.size = {len(self.w)}"
        act = sum(
            (
                self.w[feature_num]
                * (
                    x[feature_num]
                    if isinstance(x[feature_num], Value)
                    else Value(x[feature_num], label=f"e{example_idx}x{feature_num}")
                )
                for feature_num in range(len(x))
            ),
            self.b,
        )
        out = Value.activations[activation_fn](act)
        return out

    def forward(
        self,
        x: list[Value],
        example_idx: int,
        activation_fn: str = "linear",
    ) -> Value:
        return self(x, example_idx, activation_fn)

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        layer_idx: int,
        activation_fn: str = "linear",
        **kwargs,
    ):
        self.activation_fn = activation_fn
        self.layer_idx = layer_idx
        self.neurons = [
            Neuron(num_inputs, layer_idx, neuron_idx, *kwargs)
            for neuron_idx in range(num_neurons)
        ]

    def __call__(self, x: list[Value], example_idx: int) -> list[Value]:
        outs = [neuron(x, example_idx, self.activation_fn) for neuron in self.neurons]
        return outs

    def forward(self, x: list[Value], example_idx: int) -> list[Value]:
        return self(x, example_idx)

    def parameters(self) -> list[Value]:
        return [
            neuron_param
            for neuron in self.neurons
            for neuron_param in neuron.parameters()
        ]


class MLP:
    def __init__(
        self,
        num_inputs: int,
        layer_sizes: list[int],
        activation_fn: str | list[str] = "linear",
        **kwargs,
    ):
        if isinstance(activation_fn, str):
            activation_fn = [activation_fn] * len(layer_sizes)

        assert len(layer_sizes) == len(
            activation_fn
        ), f"Number of layers must be equal to number of activation functions, got {len(layer_sizes)} layers and {len(activation_fn)} activation functions"
        all_layers = [num_inputs] + layer_sizes
        self.layers = [
            Layer(
                all_layers[layer_idx],
                all_layers[layer_idx + 1],
                layer_idx,
                activation_fn[layer_idx],
                *kwargs,
            )
            for layer_idx in range(len(layer_sizes))
        ]

    def __call__(self, x: list[Value], example_num: int) -> list[Value]:
        out = x
        for layer in self.layers:
            out = layer(out, example_num)
        return out

    def forward(self, x: list[Value], example_num: int) -> list[Value]:
        return self(x, example_num)

    def parameters(self) -> list[Value]:
        return [
            layer_params for layer in self.layers for layer_params in layer.parameters()
        ]
