from __future__ import annotations

from builtins import all as py_all
from builtins import any as py_any
from typing import TYPE_CHECKING

import minidiff as md
from minidiff.topology import FuncNode

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, ParamSpec, Sequence

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
def op_func(
    **kwargs,
) -> Callable[[Callable[P, md.Tensor]], Callable[P, md.Tensor]]:
    def wrapper(func: Callable[P, md.Tensor]) -> Callable[P, md.Tensor]:
        return create_stateless_op_func(forward_func=func, **kwargs)

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
        # allow_grad = py_any([isinstance(x, md.Tensor) and x.allow_grad for x in args])
        allow_grad = False
        for x in args:
            if isinstance(x, md.Tensor) and x.allow_grad:
                allow_grad = True
                break

        wrapped_args = [x._data if isinstance(x, md.Tensor) else x for x in args]
        # print([x._data if isinstance(x, md.Tensor) else x for x in args])
        # print([x._data.dtype if isinstance(x, md.Tensor) else x for x in args])
        wrapped_kwargs = {
            key: (val._data if isinstance(val, md.Tensor) else val)
            for key, val in kwargs.items()
        }

        output = func(*wrapped_args, **wrapped_kwargs)
        # print(func)
        # print(output.dtype)
        # print(output)
        as_tensor = md.Tensor(output, allow_grad=allow_grad)

        return as_tensor

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = (
        func.__qualname__ if hasattr(func, "__qualname__") else func.__name__
    )

    return wrapper


def create_op_func(
    op_class: Callable[P, OpClass],
    is_differentiable: bool = True,
    tensor_only: bool = False,
    op_name: Optional[str] = None,
) -> Callable[P, md.Tensor]:
    if op_name is None:
        op_name = op_class.__name__

    # this is the actual op function create_op_func returns
    def minidiff_func(*op_inputs: P.args, **forward_kwargs: P.kwargs) -> md.Tensor:
        input_is_tensor = [isinstance(x, md.Tensor) for x in op_inputs]

        if not tensor_only and not py_any(input_is_tensor):
            raise ValueError(
                "This function requires at least one minidiff Tensor argument"
            )

        if tensor_only and not py_all(input_is_tensor):
            raise ValueError("This function only supports minidiff Tensors")

        # allow gradient tracking if at least one of the input tensors allows a gradient
        allow_grad = False
        for x in op_inputs:
            if isinstance(x, md.Tensor) and x.allow_grad:
                allow_grad = True
                break
        # allow_grad = py_any(
        #     [isinstance(x, md.Tensor) and x.allow_grad for x in op_inputs]
        # )

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
                grad_funcs = [None for _ in range(len(grad_funcs))]

            func_node = FuncNode(
                op_inputs=op_inputs,
                grad_functions=grad_funcs,
            )
            func_node.op_name = op_name

            output._func_node = func_node
            output.graphed = True
            for op_input in op_inputs:
                if isinstance(op_input, md.Tensor):
                    op_input.graphed = True

        return output

    minidiff_func.__name__ = op_name
    minidiff_func.__qualname__ = f"<op func '{op_name}'>"

    return minidiff_func


def _wrap_grad_funcs(grad_funcs, func_args, backward_kwargs):
    # wrapped_grad_funcs = [None] * len(grad_funcs)
    # for i, grad_func in enumerate(grad_funcs):
    #     if grad_func is None:
    #         continue
    #     wrapped_grad_funcs[i] = lambda grad: grad_func(
    #         *func_args, grad, **backward_kwargs
    #     )

    # return wrapped_grad_funcs
    def make_wrapped(grad_func):
        def wrapped_func(grad):
            return grad_func(*func_args, grad, **backward_kwargs)

        return wrapped_func

    wrapped_grads = [
        None if grad_func is None else make_wrapped(grad_func)
        for grad_func in grad_funcs
    ]
    return wrapped_grads


# for ops who don't need to be a class (i.e. don't manage their own state)
def create_stateless_op_func(
    forward_func: Callable[P, md.Tensor],
    grad_funcs: Sequence[Optional[mdt.GenericOpGrad]],
    propagate_kwargs: bool = False,
    **kwargs,
) -> Callable[P, md.Tensor]:
    class StatelessOpClass(OpClass):
        def __init__(self, *func_args, **func_kwargs):
            self.func_args = func_args
            self.func_kwargs = func_kwargs

        def create_forward(self) -> mdt.GenericFunc:
            def forward():
                return forward_func(*self.func_args, **self.func_kwargs)

            return forward

        def create_grads(self) -> Sequence[Optional[mdt.GenericOpGrad]]:
            backward_kwargs = self.func_kwargs if propagate_kwargs else {}

            # stateless op grads need inputs, grads, and kwargs, but the internal engine only provides grads
            # so create a wrapper that just automatically feeds those stored inputs and kwargs alongside grads
            # def make_wrapped(grad_func):
            #     if grad_func is None:
            #         return None

            #     def wrapped_func(grad):
            #         return grad_func(*self.func_args, grad, **backward_kwargs)

            #     return wrapped_func

            # wrapped_grads = [make_wrapped(grad_func) for grad_func in grad_funcs]
            wrapped_grad_funcs = _wrap_grad_funcs(
                grad_funcs, self.func_args, backward_kwargs
            )

            return wrapped_grad_funcs

    if "op_name" not in kwargs:
        kwargs = dict(kwargs, op_name=forward_func.__name__)

    return create_op_func(op_class=StatelessOpClass, **kwargs)


# single argument
def create_unary_op_func(
    forward_func: Callable[P, md.Tensor],
    grad: Optional[mdt.UnaryOpGrad] = None,
    **kwargs,
) -> Callable[P, md.Tensor]:
    kwargs = dict(kwargs, tensor_only=True)
    return create_stateless_op_func(
        forward_func=forward_func, grad_funcs=[grad], **kwargs
    )


# two arguments
def create_binary_op_func(
    forward_func: Callable[P, md.Tensor],
    grad_a: Optional[mdt.BinaryOpGrad] = None,
    grad_b: Optional[mdt.BinaryOpGrad] = None,
    **kwargs,
) -> Callable[P, md.Tensor]:
    return create_stateless_op_func(
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
    return create_stateless_op_func(
        forward_func=forward_func, grad_funcs=[grad_a, grad_b, grad_c], **kwargs
    )


__all__ = [
    "OpClass",
    "UnaryOpClass",
    "BinaryOpClass",
    "TernaryOpClass",
    "op_func",
    "unary_op_func",
    "binary_op_func",
    "ternary_op_func",
    "as_minidiff",
    "create_op_func",
    "create_stateless_op_func",
    "create_unary_op_func",
    "create_binary_op_func",
    "create_ternary_op_func",
]
