from __future__ import annotations

from builtins import all as py_all
from builtins import any as py_any
from typing import TYPE_CHECKING

import minidiff as md
from minidiff.topology import FuncNode

if TYPE_CHECKING:
    from typing import Callable, Optional, Sequence, List, Any, ParamSpec

    P = ParamSpec("P")

    import minidiff.typing as mdt


class OpClass:
    def create_forward(self) -> mdt.GenericFunc:
        raise NotImplementedError

    def create_grads(self) -> Sequence[Optional[mdt.GenericOpGrad]]:
        raise NotImplementedError


class UnaryOpClass(OpClass):
    def create_forward(self) -> mdt.UnaryFunc:
        raise NotImplementedError

    def create_grads(self) -> Sequence[Optional[mdt.UnaryOpGrad]]:
        raise NotImplementedError


class BinaryOpClass(OpClass):
    def create_forward(self) -> mdt.BinaryFunc:
        raise NotImplementedError

    def create_grads(self) -> Sequence[Optional[mdt.BinaryOpGrad]]:
        raise NotImplementedError


class TernaryOpClass(OpClass):
    def create_forward(self) -> mdt.TernaryFunc:
        raise NotImplementedError

    def create_grads(self) -> Sequence[Optional[mdt.TernaryOpGrad]]:
        raise NotImplementedError


# decorators which just convert a generic function to an op
def stateless_op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return generate_stateless_op_func(forward_func=func, **kwargs)

    return wrapper


def unary_op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return generate_unary_op_func(forward_func=func, **kwargs)

    return wrapper


def binary_op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return generate_binary_op_func(forward_func=func, **kwargs)

    return wrapper


def ternary_op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return generate_ternary_op_func(forward_func=func, **kwargs)

    return wrapper


# the generator functions expect Tensor functions, this just turns numpy-like to Tensor
def as_minidiff(func: Callable[..., Any]) -> Callable[..., md.Tensor]:
    def wrapper(*args, **kwargs):
        allow_grad = py_any([isinstance(x, md.Tensor) and x.allow_grad for x in args])
        wrapped_args = [x._data if isinstance(x, md.Tensor) else x for x in args]
        wrapped_kwargs = {
            key: (val._data if isinstance(val, md.Tensor) else val)
            for key, val in kwargs.items()
        }

        output = func(*wrapped_args, **wrapped_kwargs)
        as_tensor = md.Tensor(output, allow_grad=allow_grad)

        return as_tensor

    wrapper.__name__ = func.__name__

    return wrapper


def generate_op_func(
    op_class: Callable[P, None],
    is_differentiable: bool = True,
    tensor_only: bool = False,
    op_name: Optional[str] = None,
) -> Callable[P, md.Tensor]:
    # just sets the func_node property of op_output to the correct FuncNode
    def create_func_node(
        grad_funcs: List[mdt.GenericOpGrad], op_inputs: List[Any], op_name: str
    ) -> md.FuncNode:
        grads_allowed = [isinstance(x, md.Tensor) and x.allow_grad for x in op_inputs]
        # obviously tensors who don't want their gradients to be checked have no gradient function
        filtered_grad_funcs = [
            grad_func if grad_allowed else None
            for grad_func, grad_allowed in zip(grad_funcs, grads_allowed)
        ]

        func_node = FuncNode(
            op_inputs=op_inputs,
            grad_functions=filtered_grad_funcs,
        )
        func_node.op_name = op_name

        return func_node

    # this is the actual op function generate_op_func returns
    def minidiff_func(*op_inputs: P.args, **forward_kwargs: P.kwargs) -> md.Tensor:
        input_is_tensor = [isinstance(x, md.Tensor) for x in op_inputs]

        if not tensor_only and not py_any(input_is_tensor):
            raise ValueError(
                "This function requires at least one minidiff Tensor argument"
            )

        if tensor_only and not py_all(input_is_tensor):
            raise ValueError("This function only supports minidiff Tensors")

        # allow gradient tracking if at least one of the input tensors allows a gradient
        allow_grad = py_any(
            [isinstance(x, md.Tensor) and x.allow_grad for x in op_inputs]
        )

        instance = op_class(*op_inputs, **forward_kwargs)
        forward_func = instance.create_forward()

        output = forward_func()
        output.allow_grad = allow_grad

        # only attach a node if we're allowed to track gradients right now, and the tensor wants to track its gradient
        if md.grad_allowed_() and allow_grad:
            grad_funcs = instance.create_grads()
            # if the function is not differentiable, we still want to propagate the gradient to avoid breaking the
            # graph, but it is smarter to just zero out the gradients.
            if not is_differentiable:
                grad_funcs = [
                    lambda grad: md.zeros_like(grad) for _ in range(len(grad_funcs))
                ]

            func_node = create_func_node(
                grad_funcs=grad_funcs,
                op_inputs=op_inputs,
                op_name=forward_func.__name__ if op_name is None else op_name,
            )

            output.func_node = func_node
            output.graphed = True

            for op_input in op_inputs:
                if isinstance(op_input, md.Tensor):
                    op_input.graphed = True

        return output

    return minidiff_func


# for ops who don't need to be a class (i.e. don't manage their own state)
def generate_stateless_op_func(
    forward_func: Callable[P, md.Tensor],
    grad_funcs: Sequence[Optional[mdt.GenericOpGrad]],
    propagate_kwargs: bool = False,
    casting: Optional[str] = "safe",
    **kwargs,
) -> Callable[P, md.Tensor]:
    class StatelessOpClass(OpClass):
        def __init__(self, *func_args, **func_kwargs):
            self.func_args = func_args
            if casting is not None:
                func_kwargs["casting"] = casting
            self.func_kwargs = func_kwargs

        def create_forward(self) -> mdt.GenericFunc:
            def forward():
                a = forward_func(*self.func_args, **self.func_kwargs)
                return a

            forward.__name__ = forward_func.__name__

            return forward

        def create_grads(self) -> Sequence[Optional[mdt.GenericOpGrad]]:
            backward_kwargs = self.func_kwargs if propagate_kwargs else {}

            # stateless op grads need inputs, grads, and kwargs, but the internal engine only provides grads
            # so create a wrapper that just automatically feeds those stored inputs and kwargs alongside grads
            def make_wrapped(grad_func):
                def wrapped_func(grad):
                    return grad_func(*self.func_args, grad, **backward_kwargs)

                return wrapped_func

            wrapped_grads = [make_wrapped(grad_func) for grad_func in grad_funcs]

            return wrapped_grads

    return generate_op_func(op_class=StatelessOpClass, **kwargs)


# single argument
def generate_unary_op_func(
    forward_func: Callable[P, md.Tensor],
    grad: Optional[mdt.UnaryOpGrad] = None,
    **kwargs,
) -> Callable[P, md.Tensor]:
    kwargs["tensor_only"] = True
    return generate_stateless_op_func(
        forward_func=forward_func, grad_funcs=[grad], **kwargs
    )


# two arguments
def generate_binary_op_func(
    forward_func: Callable[P, md.Tensor],
    grad_a: Optional[mdt.BinaryOpGrad] = None,
    grad_b: Optional[mdt.BinaryOpGrad] = None,
    **kwargs,
) -> Callable[P, md.Tensor]:
    return generate_stateless_op_func(
        forward_func=forward_func, grad_funcs=[grad_a, grad_b], **kwargs
    )


# three arguments
def generate_ternary_op_func(
    forward_func: Callable[P, md.Tensor],
    grad_a: Optional[mdt.TernaryOpGrad] = None,
    grad_b: Optional[mdt.TernaryOpGrad] = None,
    grad_c: Optional[mdt.TernaryOpGrad] = None,
    **kwargs,
) -> Callable[P, md.Tensor]:
    return generate_stateless_op_func(
        forward_func=forward_func, grad_funcs=[grad_a, grad_b, grad_c], **kwargs
    )


__all__ = [
    "OpClass",
    "UnaryOpClass",
    "BinaryOpClass",
    "TernaryOpClass",
    "stateless_op_func",
    "unary_op_func",
    "binary_op_func",
    "ternary_op_func",
    "as_minidiff",
    "generate_op_func",
    "generate_stateless_op_func",
    "generate_unary_op_func",
    "generate_binary_op_func",
    "generate_ternary_op_func",
]
