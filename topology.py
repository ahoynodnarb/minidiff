# IMPLEMENT INHERITABLE NODE TYPES (UNARY, BINARY, etc.)
# just force user to specify derivative

from tensor import Tensor
import minidiff

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np


class FuncNode:
    def __init__(self, *inputs):
        self.inputs = inputs
        self.input_tensors = [x for x in inputs if isinstance(x, Tensor)]
        self.input_nodes = [x.func_node for x in self.input_tensors]

    def update_grads(self, grad):
        raise NotImplementedError

    def pretty_repr(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__} ({', '.join([str(x) for x in self.inputs])})"


class UnaryNode(FuncNode):
    def __init__(self, a, grad_a):
        super().__init__(a)
        self.grad_a = grad_a

    def update_grads(self, grad):
        with minidiff.no_grad():
            if self.grad_a is not None:
                a = self.inputs[0]
                a.grad += self.grad_a(a, grad)


class BinaryNode(FuncNode):
    def __init__(self, a, b, grad_a, grad_b):
        super().__init__(a, b)
        self.grad_a = grad_a
        self.grad_b = grad_b

    def update_grads(self, grad):
        a = self.inputs[0]
        b = self.inputs[1]
        with minidiff.no_grad():
            if self.grad_a is not None:
                a.grad += self.grad_a(a, b, grad)
            if self.grad_b is not None:
                b.grad += self.grad_b(a, b, grad)
