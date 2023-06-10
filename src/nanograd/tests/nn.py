# Tests for NN interface
# Written with GitHub Copilot

import unittest

from src.nanograd.nn import Neuron, Layer, MLP
from src.nanograd.value import Value


class TestNeuron(unittest.TestCase):
    def test_init(self):
        Neuron(2, 0, 0, initialization="normal")

    def test_forward(self):
        data = [Value(1), Value(1)]
        x = Neuron(2, 0, 0)
        x.forward(data, 0)

    def test_weights(self):
        x = Neuron(2, 0, 0)
        self.assertEqual(len(x.w), 2)
        self.assertIsInstance(x.b, Value)


class TestLayer(unittest.TestCase):
    def test_init(self):
        x = Layer(2, 3, 0, "relu")
        self.assertEqual(len(x.neurons), 3)
        self.assertEqual(x.activation_fn, "relu")
        self.assertEqual(x.layer_idx, 0)

    def test_forward(self):
        data = [Value(1), Value(1)]
        x = Layer(2, 3, 0)
        x.forward(data, 0)


class TestMLP(unittest.TestCase):
    def test_init(self):
        x = MLP(2, [3, 4], ["relu", "sigmoid"])
        self.assertEqual(len(x.layers), 2)

    def test_forward(self):
        data = [Value(1), Value(1)]
        x = MLP(2, [3, 4])
        x.forward(data, 0)


if __name__ == "__main__":
    unittest.main()
