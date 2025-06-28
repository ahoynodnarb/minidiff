from __future__ import annotations

from typing import TYPE_CHECKING

import minidiff as md

if TYPE_CHECKING:
    from typing import Any, List, Optional

    import minidiff.typing as mdt


class FuncNode:
    def __init__(
        self,
        op_inputs: List[Any],
        grad_functions: List[Optional[mdt.GenericOpGrad]],
    ):
        self.op_inputs = op_inputs
        self.input_tensors = [x for x in self.op_inputs if isinstance(x, md.Tensor)]
        self.grad_functions = grad_functions

        self.op_name = None

    # this accumulates gradients for the input tensors through chain rule (reverse-mode)
    def update_grads(self, grad: md.Tensor):
        # don't use no_grad() here because we are assuming gradients already don't track their gradients,
        # and if they do, they may be doing higher-order partial derivatives
        for op_input, grad_function in zip(self.op_inputs, self.grad_functions):
            if not isinstance(op_input, md.Tensor):
                continue
            if not op_input.allow_grad:
                continue
            if grad_function is None:
                continue
            grad_computation = grad_function(grad)
            # if broadcasting occured during the forward pass, we need to collect gradients
            # back in the backward pass so that the gradients are correctly distributed
            collected_grad = md.unbroadcast(grad_computation, op_input.shape)
            if op_input.grad is None:
                op_input.grad = collected_grad
            else:
                op_input.grad = op_input.grad + collected_grad

    def __repr__(self) -> str:
        return f"{self.op_name}({', '.join([str(x) for x in self.op_inputs])})"


__all__ = ["FuncNode"]
