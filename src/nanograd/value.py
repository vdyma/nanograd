from __future__ import annotations

import math

from src.nanograd.value_interface import ValueInterface

from src.nanograd.visualize import draw_dot


class Value(ValueInterface):
    activations = {
        "linear": lambda x: x,
        "tanh": lambda x: x.tanh(),
        "relu": lambda x: x.relu(),
        "sigmoid": lambda x: x.sigmoid(),
        "log": lambda x: x.log(),
    }

    def __init__(
        self,
        data: float | int,
        children: tuple[Value] = (),
        operator: str = "",
        label: str = "",
    ):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.children = set(children)
        self.operator = operator
        self.label = label

    def __repr__(self: Value) -> str:
        return f"Value(data={self.data})"

    def __neg__(self) -> Value:
        result = self * -1
        result.label = f"-{self.label}"
        return result

    def __add__(self, other: Value | float | int) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data + other.data,
            (self, other),
            "+",
            f"({self.label} + {other.label})",
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | float | int) -> Value:
        return self.__add__(other)

    def __sub__(self, other: Value | float | int) -> Value:
        result = self + (-other)
        result.label = (
            f"({self.label} - {other.label if isinstance(other, Value) else other})"
        )
        return result

    def __rsub__(self, other: Value | float | int) -> Value:
        return self.__sub__(other)

    def __mul__(self, other: Value | float | int) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data * other.data,
            (self, other),
            "*",
            f"({self.label} * {other.label})",
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | float | int) -> Value:
        return self.__mul__(other)

    def __truediv__(self, other: Value | float | int) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        result = self * other**-1
        result.label = f"({self.label} / {other.label})"
        return result

    def __pow__(self, other: float | int) -> Value:
        assert isinstance(other, (float, int)), "Exponent must be a scalar"
        out = Value(
            self.data**other, (self,), f"**{other}", f"({self.label} ** {other})"
        )

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __rpow__(self, other: float | int) -> Value:
        assert isinstance(other, (float, int)), "Exponent must be a scalar"
        out = Value(
            other**self.data, (self,), f"{other}**", f"({other} ** {self.label})"
        )

        def _backward():
            self.grad += (other**self.data) * math.log(other) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh", f"tanh({self.label})")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp", f"exp({self.label})")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0, self.data), (self,), "relu", f"relu({self.label})")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        out = Value(
            1 / (1 + math.exp(-self.data)), (self,), "sigmoid", f"sigmoid({self.label})"
        )

        def _backward():
            self.grad += (1 - out.data) * out.data * out.grad

        out._backward = _backward
        return out

    def log(self, base: float | int) -> Value:
        assert self.data > 0, "Logarithm of negative number is undefined"
        assert base > 0, "Logarithm base must be positive"
        assert base != 1, "Logarithm base cannot be 1"
        assert isinstance(base, (float, int)), "Logarithm base must be a scalar"
        out = Value(math.log(self.data, base), (self,), "log", f"log({self.label})")

        def _backward():
            self.grad += (
                1 / ((self.data * math.log(base)) if base != math.e else self.data)
            ) * out.grad

        out._backward = _backward
        return out

    def linear(self) -> Value:
        return self

    def cos(self) -> Value:
        out = Value(math.cos(self.data), (self,), "cos", f"cos({self.label})")

        def _backward():
            self.grad += -math.sin(self.data) * out.grad

        out._backward = _backward
        return out

    def sin(self) -> Value:
        out = Value(math.sin(self.data), (self,), "sin", f"sin({self.label})")

        def _backward():
            self.grad += math.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = []
        visited = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def visualize(self):
        return draw_dot(self)
