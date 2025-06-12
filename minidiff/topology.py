import minidiff as md

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np


class FuncNode:
    def __init__(self, output_tensor, input_tensors, grad_functions):
        if not isinstance(output_tensor, md.Tensor) or not all(
            [isinstance(x, md.Tensor) for x in input_tensors]
        ):
            raise ValueError("FuncNodes can only track tensors")

        self.input_tensors = (
            input_tensors if isinstance(input_tensors, list) else [input_tensors]
        )
        self.grad_functions = (
            grad_functions if isinstance(grad_functions, list) else [grad_functions]
        )
        self.input_nodes = [x.func_node for x in self.input_tensors]

        self.kwargs = {}
        self.op_name = None

        output_tensor.func_node = self
        output_tensor.graphed = True

        for tensor in input_tensors:
            tensor.graphed = True

    def update_grads(self, grad):
        # we also need to reshape/collect gradients in the case that inputs were broadcasted during the forward pass
        # don't use no_grad() here because we are assuming gradients already don't track their gradients,
        # and if they do, they may be doing higher-order partial derivatives
        for input_tensor, grad_function in zip(self.input_tensors, self.grad_functions):
            if not input_tensor.allow_grad:
                continue
            grad_computation = grad_function(*self.input_tensors, grad, **self.kwargs)
            if input_tensor.grad is None:
                input_tensor.grad = grad_computation
            else:
                input_tensor.grad += grad_computation

    def __repr__(self):
        return f"{self.op_name}({', '.join([str(x) for x in self.input_tensors])})"
