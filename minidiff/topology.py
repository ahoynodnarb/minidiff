from typing import List, Any

import minidiff as md
import minidiff.typing as mdt


class FuncNode:
    def __init__(
        self,
        op_output: md.Tensor,
        op_inputs: List[Any],
        grad_functions: List[mdt.GenericOpGrad],
    ):
        if not isinstance(op_output, md.Tensor):
            raise ValueError("FuncNodes can only track tensors")

        self.op_inputs = op_inputs
        self.input_tensors = [
            x if isinstance(x, md.Tensor) else md.Tensor(x) for x in self.op_inputs
        ]
        self.grad_functions = grad_functions

        self.kwargs = {}
        self.op_name = None

        op_output.func_node = self
        op_output.graphed = True

        for op_input in op_inputs:
            if isinstance(op_input, md.Tensor):
                op_input.graphed = True

    def update_grads(self, grad: md.Tensor):
        # don't use no_grad() here because we are assuming gradients already don't track their gradients,
        # and if they do, they may be doing higher-order partial derivatives
        for op_input, grad_function in zip(self.op_inputs, self.grad_functions):
            if not isinstance(op_input, md.Tensor):
                continue
            if not op_input.allow_grad:
                continue
            grad_computation = grad_function(*self.op_inputs, grad, **self.kwargs)
            # if broadcasting occured during the forward pass, we need to collect gradients
            # back in the backward pass so that the gradients are correctly distributed
            collected_grad = md.collect_gradients(
                grad=grad_computation, target_shape=op_input.shape
            )
            if op_input.grad is None:
                op_input.grad = collected_grad
            else:
                op_input.grad += collected_grad

    def __repr__(self) -> str:
        return f"{self.op_name}({', '.join([str(x) for x in self.op_inputs])})"
