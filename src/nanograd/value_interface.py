# Abstract class for value interface
from __future__ import annotations

from abc import ABC, abstractmethod


class ValueInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        data: float | int,
        children: tuple[ValueInterface, ValueInterface]
        | tuple[ValueInterface]
        | tuple[()] = (),
        operator: str = "",
        label: str = "",
    ):
        self.data = data
        self.grad = 0
        self.children = children or []
        self.operator = operator
        self.label = label
