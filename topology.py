import minidiff as md

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np


class FuncNode:
    def __init__(self, *inputs):
        self.inputs = inputs
        self.input_tensors = [x for x in inputs if isinstance(x, md.Tensor)]
        self.input_nodes = [x.func_node for x in self.input_tensors]
        self.kwargs = {}

    def update_grads(self, grad):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__} ({', '.join([str(x) for x in self.inputs])})"


class UnaryNode(FuncNode):
    def __init__(self, a, grad_a):
        super().__init__(a)
        a.graphed = True
        self.grad_a = grad_a

    def update_grads(self, grad):
        a = self.inputs[0]
        with md.no_grad():
            if self.grad_a is not None and a.allow_grad:
                a.grad += self.grad_a(a, grad, **self.kwargs)


class BinaryNode(FuncNode):
    def __init__(self, a, b, grad_a, grad_b):
        super().__init__(a, b)
        if isinstance(a, md.Tensor):
            a.graphed = True
        if isinstance(b, md.Tensor):
            b.graphed = True
        self.grad_a = grad_a
        self.grad_b = grad_b

    def update_grads(self, grad):
        a = self.inputs[0]
        b = self.inputs[1]
        with md.no_grad():
            if self.grad_a is not None and a.allow_grad:
                a.grad += self.grad_a(a, b, grad, **self.kwargs)
            if self.grad_b is not None and b.allow_grad:
                b.grad += self.grad_b(a, b, grad, **self.kwargs)
