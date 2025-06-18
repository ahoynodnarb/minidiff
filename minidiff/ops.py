from builtins import all as py_all, any as py_any
from typing import Tuple, Optional, Type, Sequence, Union, Any, Dict

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import minidiff as md
from minidiff.topology import FuncNode
import minidiff.typing as mdt
from minidiff.utils import get_exported_var_names


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
def stateless_op_func(**kwargs) -> mdt.GenericOp:
    def wrapper(func):
        return generate_stateless_op_func(forward_func=func, **kwargs)

    return wrapper


def unary_op_func(**kwargs) -> mdt.UnaryOp:
    def wrapper(func):
        return generate_unary_op_func(forward_func=func, **kwargs)

    return wrapper


def binary_op_func(**kwargs) -> mdt.BinaryOp:
    def wrapper(func):
        return generate_binary_op_func(forward_func=func, **kwargs)

    return wrapper


def ternary_op_func(**kwargs) -> mdt.TernaryOp:
    def wrapper(func):
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


class TensorDot(BinaryOpClass):

    def create_forward(self) -> mdt.BinaryOp:
        return np.tensordot

    def create_grads(self) -> Tuple[mdt.BinaryOpGrad, mdt.BinaryOpGrad]:

        def grad_a(
            a: md.Tensor,
            b: md.Tensor,
            grad: md.Tensor,
            axes: Union[int, mdt.NestedSequence[int]] = 2,
        ) -> md.Tensor:
            if isinstance(axes, int):
                axes_a = tuple(range(a.ndim - axes, a.ndim))
                axes_b = tuple(range(axes))
                axes = (axes_a, axes_b)
            # indices of all dims in b not originally contracted in the forward tensordot
            uncontracted_a = tuple(i for i in range(a.ndim) if i not in axes[0])
            uncontracted_b = tuple(i for i in range(b.ndim) if i not in axes[1])
            # indices of all dims in grad that align with uncontracted_b
            grad_aligned = tuple(range(grad.ndim - len(uncontracted_b), grad.ndim))
            new_axes = (grad_aligned, uncontracted_b)
            result = tensordot(grad, b, axes=new_axes)
            # first few indices will be uncontracted in a, last few will be contracted in a (original forward pass)
            # need to transpose such that the first few take up the uncontracted a indices, and the last few take up the contracted a indices
            permutation_indices = [0] * a.ndim
            n_uncontracted_a = len(uncontracted_a)
            uncontracted_idx = 0
            contracted_idx = 0
            for i in range(a.ndim):
                if i < n_uncontracted_a:
                    permutation_indices[uncontracted_a[uncontracted_idx]] = i
                    uncontracted_idx += 1
                else:
                    permutation_indices[axes[0][contracted_idx]] = i
                    contracted_idx += 1

            reshaped = md.transpose(result, permutation_indices)
            return reshaped

        def grad_b(
            a: md.Tensor,
            b: md.Tensor,
            grad: md.Tensor,
            axes: Union[int, mdt.NestedSequence[int]] = 2,
        ) -> md.Tensor:
            if isinstance(axes, int):
                axes_a = tuple(range(a.ndim - axes, a.ndim))
                axes_b = tuple(range(axes))
                axes = (axes_a, axes_b)
            # indices of all dims in a not originally contracted in the forward tensordot
            uncontracted_a = tuple(i for i in range(a.ndim) if i not in axes[0])
            uncontracted_b = tuple(i for i in range(b.ndim) if i not in axes[1])
            # indices of all dims in grad that align with uncontracted_a
            grad_aligned = tuple(range(len(uncontracted_a)))
            new_axes = (uncontracted_a, grad_aligned)
            result = tensordot(a, grad, axes=new_axes)
            # first few indices of result will be contracted in a, last few will be uncontracted in b (original forward pass
            # need to transpose so that the last few take up the uncontracted b indices, and the first few take up the original contracted indices
            n_contracted_a = len(axes[0])
            contracted_idx = 0
            uncontracted_idx = 0
            permutation_indices = [0] * b.ndim
            for i in range(b.ndim):
                if i < n_contracted_a:
                    permutation_indices[axes[1][contracted_idx]] = i
                    contracted_idx += 1
                else:
                    permutation_indices[uncontracted_b[uncontracted_idx]] = i
                    uncontracted_idx += 1

            reshaped = md.transpose(result, permutation_indices)
            return reshaped

        return (grad_a, grad_b)


def s__grad(a: md.Tensor, key: Any, grad: md.Tensor) -> md.Tensor:
    ret = md.zeros_like(a)
    np.add.at(ret._data, key, grad._data)
    return ret


exported_ops = [
    transpose := generate_binary_op_func(
        forward_func=np.transpose,
        grad_a=lambda a, grad, axes=None: transpose(grad, axes=axes),
        is_backend_op=True,
        propagate_kwargs=True,
        casting=None,
    ),
    swapaxes := generate_ternary_op_func(
        forward_func=np.swapaxes,
        grad_a=lambda a, axis1, axis2, grad, **kwargs: swapaxes(
            grad, axis1, axis2, **kwargs
        ),
        is_backend_op=True,
        propagate_kwargs=True,
        casting=None,
    ),
    flip := generate_unary_op_func(
        forward_func=np.flip,
        grad=lambda a, grad, **kwargs: flip(grad, **kwargs),
        is_backend_op=True,
        propagate_kwargs=True,
        casting=None,
    ),
    broadcast_to := generate_binary_op_func(
        forward_func=np.broadcast_to,
        grad_a=lambda a, grad: md.collect_gradients(grad=grad, target_shape=a.shape),
        is_backend_op=True,
        casting=None,
    ),
    atleast_1d := generate_unary_op_func(
        forward_func=np.atleast_1d,
        grad=lambda a, grad: grad,
        is_backend_op=True,
        casting=None,
    ),
    atleast_2d := generate_unary_op_func(
        forward_func=np.atleast_2d,
        grad=lambda a, grad: grad,
        is_backend_op=True,
        casting=None,
    ),
    atleast_3d := generate_unary_op_func(
        forward_func=np.atleast_3d,
        grad=lambda a, grad: grad,
        is_backend_op=True,
        casting=None,
    ),
    copy := generate_binary_op_func(
        forward_func=np.copy,
        grad_a=lambda a, grad: grad,
        is_backend_op=True,
        casting=None,
    ),
    s_ := generate_binary_op_func(
        forward_func=lambda a, key: a[key],
        grad_a=s__grad,
        grad_b=None,
        is_backend_op=True,
        casting=None,
        op_name="index",
    ),
    clip := generate_unary_op_func(
        forward_func=np.clip,
        grad=lambda a, grad, a_min=None, a_max=None: grad
        * logical_and(
            a > float("-inf") if a_min is None else a_min,
            a < float("inf") if a_max is None else a_max,
        ),
        propagate_kwargs=True,
        is_backend_op=True,
        casting=None,
    ),
    reshape := generate_binary_op_func(
        forward_func=np.reshape,
        grad_a=lambda a, b, grad: grad.reshape(a.shape),
        grad_b=None,
        is_backend_op=True,
        casting=None,
    ),
    matmul := generate_binary_op_func(
        forward_func=np.matmul,
        grad_a=lambda a, b, grad: matmul(grad, b.t),
        grad_b=lambda a, b, grad: matmul(a.t, grad),
        tensor_only=True,
        is_backend_op=True,
        casting=None,
    ),
    tensordot := generate_op_func(
        op_class=TensorDot,
        tensor_only=True,
        is_backend_op=True,
        propagate_kwargs=True,
        casting=None,
    ),
    add := generate_binary_op_func(
        forward_func=np.add,
        grad_a=lambda a, b, grad: grad,
        grad_b=lambda a, b, grad: grad,
        is_backend_op=True,
    ),
    subtract := generate_binary_op_func(
        forward_func=np.subtract,
        grad_a=lambda a, b, grad: grad,
        grad_b=lambda a, b, grad: -grad,
        is_backend_op=True,
    ),
    multiply := generate_binary_op_func(
        forward_func=np.multiply,
        grad_a=lambda a, b, grad: grad * b,
        grad_b=lambda a, b, grad: grad * a,
        is_backend_op=True,
    ),
    true_divide := generate_binary_op_func(
        forward_func=np.true_divide,
        grad_a=lambda a, b, grad: grad / b,
        grad_b=lambda a, b, grad: (-grad * a) / (b**2),
        is_backend_op=True,
    ),
    floor_divide := generate_binary_op_func(
        forward_func=np.floor_divide, is_differentiable=False, is_backend_op=True
    ),
    power := generate_binary_op_func(
        forward_func=np.power,
        grad_a=lambda a, b, grad: grad * b * (a ** (b - 1)),
        grad_b=lambda a, b, grad: grad * log(a) * a**b,
        is_backend_op=True,
    ),
    sqrt := lambda a, b, **kwargs: power(a, 0.5, **kwargs),
    floor := generate_unary_op_func(
        forward_func=np.floor, is_differentiable=False, is_backend_op=True
    ),
    ceil := generate_unary_op_func(
        forward_func=np.ceil, is_differentiable=False, is_backend_op=True
    ),
    cos := generate_unary_op_func(
        forward_func=np.cos, grad=lambda a, grad: grad * -sin(a), is_backend_op=True
    ),
    sin := generate_unary_op_func(
        forward_func=np.sin, grad=lambda a, grad: grad * cos(a), is_backend_op=True
    ),
    tan := generate_unary_op_func(
        forward_func=np.tan,
        grad=lambda a, grad: grad * (1 / cos(a) ** 2),
        is_backend_op=True,
    ),
    cosh := generate_unary_op_func(
        forward_func=np.cosh, grad=lambda a, grad: grad * sinh(a), is_backend_op=True
    ),
    sinh := generate_unary_op_func(
        forward_func=np.sinh, grad=lambda a, grad: grad * cosh(a), is_backend_op=True
    ),
    tanh := generate_unary_op_func(
        forward_func=np.tanh,
        grad=lambda a, grad: grad * (1 / cosh(a) ** 2),
        is_backend_op=True,
    ),
    exp := generate_unary_op_func(
        forward_func=np.exp, grad=lambda a, grad: grad * exp(a), is_backend_op=True
    ),
    log := generate_unary_op_func(
        forward_func=np.log, grad=lambda a, grad: grad / a, is_backend_op=True
    ),
    sum := generate_unary_op_func(
        forward_func=np.sum, grad=lambda a, grad: grad, is_backend_op=True, casting=None
    ),
    mean := generate_unary_op_func(
        forward_func=np.mean,
        grad=lambda a, grad: grad / a.size,
        is_backend_op=True,
        casting=None,
    ),
    greater := generate_binary_op_func(
        forward_func=np.greater,
        is_differentiable=False,
        is_backend_op=True,
        casting=None,
    ),
    greater_equal := generate_binary_op_func(
        forward_func=np.greater_equal,
        is_differentiable=False,
        is_backend_op=True,
        casting=None,
    ),
    less := generate_binary_op_func(
        forward_func=np.less, is_differentiable=False, is_backend_op=True, casting=None
    ),
    less_equal := generate_binary_op_func(
        forward_func=np.less_equal,
        is_differentiable=False,
        is_backend_op=True,
        casting=None,
    ),
    equal := generate_binary_op_func(
        forward_func=np.equal, is_differentiable=False, is_backend_op=True, casting=None
    ),
    not_equal := generate_binary_op_func(
        forward_func=np.not_equal,
        is_differentiable=False,
        is_backend_op=True,
        casting=None,
    ),
    logical_and := generate_binary_op_func(
        forward_func=np.logical_and, is_differentiable=False, is_backend_op=True
    ),
    logical_or := generate_binary_op_func(
        forward_func=np.logical_or, is_differentiable=False, is_backend_op=True
    ),
    logical_not := generate_binary_op_func(
        forward_func=np.logical_not, is_differentiable=False, is_backend_op=True
    ),
    logical_xor := generate_binary_op_func(
        forward_func=np.logical_xor, is_differentiable=False, is_backend_op=True
    ),
    sign := generate_unary_op_func(
        forward_func=np.sign, is_differentiable=False, is_backend_op=True
    ),
    absolute := generate_unary_op_func(
        forward_func=np.absolute,
        grad=lambda a, grad: grad * sign(a),
        is_backend_op=True,
    ),
    all := generate_unary_op_func(
        forward_func=np.all, is_differentiable=False, is_backend_op=True, casting=None
    ),
    any := generate_unary_op_func(
        forward_func=np.any, is_differentiable=False, is_backend_op=True, casting=None
    ),
]


__all__ = get_exported_var_names(local_vars=dict(locals()), exported_vars=exported_ops)
