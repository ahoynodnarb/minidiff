import minidiff as md

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np


class FuncNode:
    def __init__(self, *inputs):
        if not all([isinstance(x, md.Tensor) for x in inputs]):
            raise ValueError("FuncNodes can only track tensors")

        self.input_tensors = [x for x in inputs if isinstance(x, md.Tensor)]
        self.input_nodes = [x.func_node for x in self.input_tensors]

        self.kwargs = {}
        self.op_name = None

    def update_grads(self, grad):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__} ({', '.join([str(x) for x in self.input_tensors])})"


class UnaryNode(FuncNode):
    def __init__(self, a, grad_a):
        super().__init__(a)
        a.graphed = True
        self.grad_a = grad_a

    def update_grads(self, grad):
        a = self.input_tensors[0]
        with md.no_grad():
            if a.allow_grad and self.grad_a is not None:
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
        a = self.input_tensors[0]
        b = self.input_tensors[1]
        with md.no_grad():
            if a.allow_grad and self.grad_a:
                a.grad += self.grad_a(a, b, grad, **self.kwargs)
            if b.allow_grad and self.grad_b:
                b.grad += self.grad_b(a, b, grad, **self.kwargs)
