from __future__ import annotations

from typing import TYPE_CHECKING

import minidiff as md
from minidiff.topology import FuncNode
from minidiff.utils import try_unwrap

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, ParamSpec, Sequence

    P = ParamSpec("P")

    import minidiff.typing as mdt


def _should_allow_grad(op_inputs: Sequence[Any]):
    for x in op_inputs:
        if isinstance(x, md.Tensor) and x.allow_grad:
            return True

    return False


def _validate_op_inputs(op_inputs: Sequence[Any], tensor_only: bool):
    success = False
    for t in op_inputs:
        is_tensor = isinstance(t, md.Tensor)

        success = is_tensor
        if success and not tensor_only:
            break
        if not success and tensor_only:
            break

    if success:
        return

    error_msg = (
        "This function only supports minidiff Tensors"
        if tensor_only
        else "This function requires at least one minidiff Tensor argument"
    )
    raise ValueError(error_msg)


# decorators which just convert a generic function to an op
def op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return create_op_func(forward_func=func, **kwargs)

    return wrapper


def unary_op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return create_unary_op_func(forward_func=func, **kwargs)

    return wrapper


def binary_op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return create_binary_op_func(forward_func=func, **kwargs)

    return wrapper


def ternary_op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return create_ternary_op_func(forward_func=func, **kwargs)

    return wrapper


# the generator functions expect Tensor functions, this just turns numpy-like to Tensor
def as_minidiff(func: Callable[..., Any]) -> Callable[..., md.Tensor]:
    def wrapper(*args, **kwargs):
        allow_grad = _should_allow_grad(args)

        wrapped_args = [try_unwrap(x) for x in args]
        wrapped_kwargs = {key: try_unwrap(val) for key, val in kwargs.items()}

        output = func(*wrapped_args, **wrapped_kwargs)
        as_tensor = md.Tensor(output, allow_grad=allow_grad)

        return as_tensor

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = (
        func.__qualname__ if hasattr(func, "__qualname__") else func.__name__
    )

    return wrapper


def create_op_func(
    forward_func: Callable[P, md.Tensor],
    grad_funcs: Sequence[Optional[mdt.GenericOpGrad]],
    propagate_kwargs: bool = False,
    is_differentiable: bool = True,
    tensor_only: bool = False,
    op_name: Optional[str] = None,
) -> Callable[P, md.Tensor]:
    # if the function is not differentiable, we still want to propagate the gradient to avoid breaking the
    # graph, but it is smarter to just zero out the gradients.
    if not is_differentiable:
        grad_funcs = [None for _ in range(len(grad_funcs))]

    if op_name is None:
        op_name = forward_func.__name__

    # this is the actual op function create_op_func returns
    def minidiff_func(*op_inputs: P.args, **op_kwargs: P.kwargs) -> md.Tensor:
        _validate_op_inputs(op_inputs, tensor_only)
        # allow gradient tracking if at least one of the input tensors allows a gradient
        allow_grad = _should_allow_grad(op_inputs)
        output = forward_func(*op_inputs, **op_kwargs)
        output.allow_grad = allow_grad

        # only attach a node if we're allowed to track gradients right now, and the tensor wants to track its gradient
        if allow_grad and md.grad_allowed_():
            # the output already is part of some graph, so we just adopt it into this one
            if output.func_node is None:
                output.func_node = FuncNode(
                    grad_functions=grad_funcs,
                    op_inputs=op_inputs,
                    op_kwargs=op_kwargs,
                    op_name=op_name,
                    propagate_kwargs=propagate_kwargs,
                )

        return output

    minidiff_func.__name__ = op_name
    minidiff_func.__qualname__ = f"<op func '{op_name}'>"

    return minidiff_func


# single argument
def create_unary_op_func(
    forward_func: Callable[P, md.Tensor],
    grad: Optional[mdt.UnaryOpGrad] = None,
    **kwargs,
) -> Callable[P, md.Tensor]:
    kwargs = dict(kwargs, tensor_only=True)
    return create_op_func(forward_func=forward_func, grad_funcs=[grad], **kwargs)


# two arguments
def create_binary_op_func(
    forward_func: Callable[P, md.Tensor],
    grad_a: Optional[mdt.BinaryOpGrad] = None,
    grad_b: Optional[mdt.BinaryOpGrad] = None,
    **kwargs,
) -> Callable[P, md.Tensor]:
    return create_op_func(
        forward_func=forward_func, grad_funcs=[grad_a, grad_b], **kwargs
    )


# three arguments
def create_ternary_op_func(
    forward_func: Callable[P, md.Tensor],
    grad_a: Optional[mdt.TernaryOpGrad] = None,
    grad_b: Optional[mdt.TernaryOpGrad] = None,
    grad_c: Optional[mdt.TernaryOpGrad] = None,
    **kwargs,
) -> Callable[P, md.Tensor]:
    return create_op_func(
        forward_func=forward_func, grad_funcs=[grad_a, grad_b, grad_c], **kwargs
    )


__all__ = [
    "op_func",
    "unary_op_func",
    "binary_op_func",
    "ternary_op_func",
    "as_minidiff",
    "create_op_func",
    "create_unary_op_func",
    "create_binary_op_func",
    "create_ternary_op_func",
]
