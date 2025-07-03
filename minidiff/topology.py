from __future__ import annotations

from typing import TYPE_CHECKING

import minidiff as md

if TYPE_CHECKING:
    from typing import Any, Dict, Optional, Sequence

    import minidiff.typing as mdt


class FuncNode:
    def __init__(
        self,
        grad_functions: Sequence[Optional[mdt.GenericOpGrad]],
        op_inputs: Sequence[Any],
        op_kwargs: Optional[Dict[str, Any]] = None,
        op_name: Optional[str] = None,
        propagate_kwargs: bool = False,
    ):
        self.grad_functions = grad_functions

        self.op_inputs = op_inputs

        if op_kwargs is None:
            op_kwargs = {}
        self.op_kwargs = op_kwargs

        if op_name is None:
            op_name = ""
        self.op_name = op_name

        self.propagate_kwargs = propagate_kwargs

        self.tensor_inputs = [x for x in op_inputs if isinstance(x, md.Tensor)]

        for tensor in self.tensor_inputs:
            tensor.graph_refs += 1

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

            kwargs = self.op_kwargs if self.propagate_kwargs else {}
            grad_computation = grad_function(*self.op_inputs, grad, **kwargs)
            # if broadcasting occured during the forward pass, we need to collect gradients
            # back in the backward pass so that the gradients are correctly distributed
            collected_grad = md.unbroadcast(grad_computation, op_input.shape)
            if op_input.grad is None:
                op_input.grad = collected_grad
            else:
                op_input.grad = op_input.grad + collected_grad

    def __repr__(self) -> str:
        return f"{self.op_name}({', '.join([str(x) for x in self.op_inputs])})"
