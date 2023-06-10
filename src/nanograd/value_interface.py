# Abstract class for value interface
# Written with GitHub Copilot

from abc import ABC, abstractmethod

class ValueInterface(ABC):
    @abstractmethod
    def __init__(self, data, grad=0, _prev=None, _op=None, label=None):
        self.data = data
        self.grad = grad
        self._prev = _prev or []
        self._op = _op
        self.label = label
