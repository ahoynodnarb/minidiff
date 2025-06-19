from builtins import any as py_any, all as py_all
from typing import Sequence, Optional, Callable, Type

import minidiff as md
from minidiff.topology import FuncNode
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
def stateless_op_func(**kwargs) -> Callable[[mdt.GenericFunc], mdt.GenericOp]:
    def wrapper(func: mdt.GenericFunc) -> mdt.GenericOp:
        return generate_stateless_op_func(forward_func=func, **kwargs)

    return wrapper


def unary_op_func(**kwargs) -> Callable[[mdt.UnaryFunc], mdt.UnaryOp]:
    def wrapper(func: mdt.UnaryFunc) -> mdt.UnaryOp:
        return generate_unary_op_func(forward_func=func, **kwargs)

    return wrapper


def binary_op_func(**kwargs) -> Callable[[mdt.BinaryFunc], mdt.BinaryOp]:
    def wrapper(func: mdt.BinaryFunc) -> mdt.BinaryOp:
        return generate_binary_op_func(forward_func=func, **kwargs)

    return wrapper


def ternary_op_func(**kwargs) -> Callable[[mdt.TernaryFunc], mdt.TernaryOp]:
    def wrapper(func: mdt.TernaryFunc) -> mdt.TernaryOp:
        return generate_ternary_op_func(forward_func=func, **kwargs)

    return wrapper


def generate_op_func(
    op_class: Type[OpClass],
    is_differentiable: bool = True,
    tensor_only: bool = False,
    is_backend_op: bool = False,
    propagate_kwargs: bool = False,
    op_name: Optional[str] = None,
    casting: Optional[str] = "safe",
) -> mdt.GenericOp:
    instance = op_class()
    forward_func = instance.create_forward()
    grad_funcs = instance.create_grads()
    # if the function is not differentiable, we still want to propagate the gradient to avoid breaking the
    # graph, but it is smarter to just zero out the gradients.
    if not is_differentiable:
        grad_funcs = [
            lambda a, b, grad: md.zeros_like(grad) for _ in range(len(grad_funcs))
        ]

    # just sets the func_node property of op_output to the correct FuncNode
    def attach_func_node(op_output, op_inputs, forward_kwargs):
        grads_allowed = [isinstance(x, md.Tensor) and x.allow_grad for x in op_inputs]
        # obviously tensors who don't want their gradients to be checked have no gradient function
        filtered_grad_funcs = [
            grad_func if grad_allowed else None
            for grad_func, grad_allowed in zip(grad_funcs, grads_allowed)
        ]

        func_node = FuncNode(
            op_output=op_output,
            op_inputs=op_inputs,
            grad_functions=filtered_grad_funcs,
        )
        func_node.op_name = forward_func.__name__ if op_name is None else op_name
        if propagate_kwargs:
            func_node.kwargs = forward_kwargs

        op_output.func_node = func_node
        op_output.graphed = True

        for op_input in op_inputs:
            if isinstance(op_input, md.Tensor):
                op_input.graphed = True

    # correctly formats forward inputs, gets the output, and casts back into a Tensor if necessary
    def get_op_output(op_inputs, allow_grad, forward_kwargs):
        # if the op is a traditional numpy function, then we need to "uncast" it back to numpy
        if is_backend_op:
            forward_inputs = [
                x._data if isinstance(x, md.Tensor) else x for x in op_inputs
            ]
        else:
            forward_inputs = op_inputs

        if casting is None:
            output = forward_func(*forward_inputs, **forward_kwargs)
        else:
            output = forward_func(*forward_inputs, casting=casting, **forward_kwargs)

        # traditional numpy functions of course return numpy objects, so we need to wrap in a Tensor
        if is_backend_op:
            output = md.Tensor(output)

        # ensure gradient tracking rules do not break
        output.allow_grad = allow_grad

        return output

    # this is the actual op function generate_op_func returns
    def minidiff_func(*op_inputs, **forward_kwargs):
        input_is_tensor = [isinstance(x, md.Tensor) for x in op_inputs]

        if not tensor_only and not py_any(input_is_tensor):
            raise ValueError(
                "minidiff functions only work when at least one argument is a minidiff Tensor"
            )

        if tensor_only and not py_all(input_is_tensor):
            raise ValueError("This function only supports minidiff Tensors")

        # allow gradient tracking if at least one of the input tensors allows a gradient
        allow_grad = py_any(
            [isinstance(x, md.Tensor) and x.allow_grad for x in op_inputs]
        )

        output = get_op_output(
            op_inputs=op_inputs, allow_grad=allow_grad, forward_kwargs=forward_kwargs
        )

        # only attach a node if we're allowed to track gradients right now, and the tensor wants to track its gradient
        if md.grad_allowed_() and allow_grad:
            attach_func_node(
                op_output=output, op_inputs=op_inputs, forward_kwargs=forward_kwargs
            )

        return output

    return minidiff_func


# for ops who don't need to be a class (i.e. don't manage their own state)
def generate_stateless_op_func(
    forward_func: mdt.GenericFunc,
    grad_funcs: Sequence[Optional[mdt.GenericOpGrad]],
    **kwargs,
) -> mdt.GenericOp:

    class StatelessOpClass(OpClass):
        def create_forward(self) -> mdt.GenericFunc:
            return forward_func

        def create_grads(self) -> Sequence[Optional[mdt.GenericOpGrad]]:
            return grad_funcs

    return generate_op_func(op_class=StatelessOpClass, **kwargs)


# single argument
def generate_unary_op_func(
    forward_func: mdt.UnaryFunc,
    grad: Optional[mdt.UnaryOpGrad] = None,
    **kwargs,
) -> mdt.UnaryOp:
    kwargs["tensor_only"] = True
    return generate_stateless_op_func(
        forward_func=forward_func, grad_funcs=[grad], **kwargs
    )


# two arguments
def generate_binary_op_func(
    forward_func: mdt.BinaryFunc,
    grad_a: Optional[mdt.BinaryOpGrad] = None,
    grad_b: Optional[mdt.BinaryOpGrad] = None,
    **kwargs,
) -> mdt.BinaryOp:
    return generate_stateless_op_func(
        forward_func=forward_func, grad_funcs=[grad_a, grad_b], **kwargs
    )


# three arguments
def generate_ternary_op_func(
    forward_func: mdt.TernaryFunc,
    grad_a: Optional[mdt.TernaryOpGrad] = None,
    grad_b: Optional[mdt.TernaryOpGrad] = None,
    grad_c: Optional[mdt.TernaryOpGrad] = None,
    **kwargs,
) -> mdt.TernaryOp:
    return generate_stateless_op_func(
        forward_func=forward_func, grad_funcs=[grad_a, grad_b, grad_c], **kwargs
    )
