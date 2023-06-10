from __future__ import annotations

import math

from src.visualize import draw_dot


class Value:
    activations = {
        'linear': lambda x: x,
        'tanh': lambda x: x.tanh(),
        'relu': lambda x: x.relu(),
        'sigmoid': lambda x: x.sigmoid(),
        'log': lambda x: x.log(),
    }

    def __init__(self, data: float | int, _children: tuple[Value]=(), _op: str='', label: str=''):
        self.data = data
        self.grad = .0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self: Value) -> str:
        return f"Value(data={self.data})"
    
    def __neg__(self) -> Value:
        return self * -1
    
    def __add__(self, other: Value | float | int) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other: Value | float | int) -> Value:
        return self.__add__(other)
    
    def __sub__(self, other: Value | float | int) -> Value:
        return self + (-other)
    
    def __rsub__(self, other: Value | float | int) -> Value:
        return self.__sub__(other)
        
    def __mul__(self, other: Value | float | int) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other: Value | float | int) -> Value:
        return self.__mul__(other)
    
    def __truediv__(self, other: Value | float | int) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1
    
    def __pow__(self, other: Value | float | int) -> Value:
        assert isinstance(other, (float, int)), "Exponent must be a scalar"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def relu(self) -> Value:
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self) -> Value:
        out = Value(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')

        def _backward():
            self.grad += (1 - out.data) * out.data * out.grad
        out._backward = _backward

        return out

    def log(self) -> Value:
        assert self.data > 0, "Logarithm of negative number is undefined"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out
    
    def backward(self) -> None:
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def visualize(self) -> None:
        return draw_dot(self)
