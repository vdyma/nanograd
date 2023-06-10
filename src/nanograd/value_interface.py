# Abstract class for value interface
# Written with GitHub Copilot

from abc import ABC, abstractmethod


class ValueInterface(ABC):
    @abstractmethod
    def __init__(self, data, grad=0, _children=[], _op="", label=""):
        self.data = data
        self.grad = grad
        self.children = _children or []
        self.operation = _op
        self.label = label
