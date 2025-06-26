from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import minidiff as md
import minidiff.ops as ops
from minidiff.utils import get_exported_var_names

if TYPE_CHECKING:
    from typing import Any, Optional, Sequence, Tuple, Union

    import minidiff.typing as mdt


def tensordot_grad_a(
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

    reshaped = md.transpose(result, axes=permutation_indices)
    return reshaped


def tensordot_grad_b(
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

    reshaped = md.transpose(result, axes=permutation_indices)
    return reshaped


def max_grad(
    a: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    max_indices = argmax(a, axis=axis)
    unraveled = md.unravel_index(max_indices, a.shape)
    ret = md.zeros_like(a)
    ret[unraveled] = grad
    return grad


def prod_grad(
    a: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    if axis == ():
        return grad
    multiplied = prod(a, axis=axis, keepdims=True)
    return where(a == 0, 0, grad * multiplied / a)


def transpose_grad(
    a: md.Tensor, grad: md.Tensor, axes: Optional[Union[int, Tuple[int]]] = None
) -> md.Tensor:
    if axes is None:
        return transpose(grad)
    backward_axes = [-1] * len(axes)
    for i, dim in enumerate(axes):
        backward_axes[dim] = i
    return transpose(grad, axes=backward_axes)


# if broadcasting happened during the forward pass, you need to correctly
# sum up the correct dimensions so that the gradients match up
def unbroadcast_forward(a: md.Tensor, target_shape: Sequence[int]) -> md.Tensor:
    # this collects the prepended axes
    # numpy inserts dimensions from the left when broadcasting
    # this just sums across those prepended dimensions
    len_prepended = a.ndim - len(target_shape)
    broadcasted_axes = tuple(range(len_prepended))
    if len(broadcasted_axes) != 0:
        a = a.sum(axis=broadcasted_axes)

    # this collects the axes that were stretched to span a greater dim
    # numpy will "stretch" dimensions so that dimensions of size 1 are tiled
    # so that they match the greater dimension.
    # we can undo this by just summing across those dimensions until we reach size 1 again
    ndims = min(len(target_shape), a.ndim)
    stretched_axes = tuple(
        i for i in range(ndims) if a.shape[i] > 1 and target_shape[i] == 1
    )
    if len(stretched_axes) != 0:
        a = a.sum(axis=stretched_axes, keepdims=True)

    # final reshape operation that can upsample if necessary
    return md.broadcast_to(a, target_shape)


def getitem_grad(a: md.Tensor, key: Any, grad: md.Tensor) -> md.Tensor:
    ret = md.zeros_like(a)
    np.add.at(ret._data, key, grad._data)
    return ret


def mean_grad(
    a: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    if axis is None:
        return grad / a.size
    if axis == ():
        return grad
    if isinstance(axis, int):
        return grad / a.shape[axis]
    in_shape = a.shape
    multiplied_dims = [in_shape[dim] for dim in axis]
    return prod(multiplied_dims)


exported_ops = [
    ravel := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(lambda a, order="C": a.ravel(order=order)),
        grad=lambda a, grad, order="C": ravel(grad, order=order),
        casting=None,
    ),
    flatten := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(lambda a, order="C": a.flatten(order=order)),
        grad=lambda a, grad, order="C": flatten(grad, order=order),
        casting=None,
    ),
    expand_dims := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.expand_dims),
        grad_a=lambda a, axis, grad: expand_dims(grad, axis),
        casting=None,
    ),
    astype := ops.generate_binary_op_func(
        forward_func=lambda a, dtype, **kwargs: md.Tensor(
            a._data.astype(dtype, **kwargs), dtype=dtype
        ),
        grad_a=lambda a, dtype, grad: grad.astype(a.dtype),
        casting=None,
    ),
    argmax := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.argmax),
        is_differentiable=False,
        casting=None,
    ),
    max := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.max),
        grad=max_grad,
        propagate_kwargs=True,
        casting=None,
    ),
    argwhere := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.argwhere),
        is_differentiable=False,
        casting=None,
    ),
    where := ops.generate_ternary_op_func(
        forward_func=ops.as_minidiff(np.where),
        grad_b=lambda condition, b, c: b * condition,
        grad_c=lambda condition, b, c: c * ~condition,
        casting=None,
    ),
    prod := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.prod),
        grad=prod_grad,
        propagate_kwargs=True,
        casting=None,
    ),
    transpose := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.transpose),
        grad=transpose_grad,
        propagate_kwargs=True,
        casting=None,
    ),
    swapaxes := ops.generate_ternary_op_func(
        forward_func=ops.as_minidiff(np.swapaxes),
        grad_a=lambda a, axis1, axis2, grad, **kwargs: swapaxes(
            grad, axis1, axis2, **kwargs
        ),
        propagate_kwargs=True,
        casting=None,
    ),
    flip := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.flip),
        grad=lambda a, grad, **kwargs: flip(grad, **kwargs),
        propagate_kwargs=True,
        casting=None,
    ),
    unbroadcast := ops.generate_binary_op_func(
        forward_func=unbroadcast_forward,
        grad_a=lambda a, shape, grad: broadcast_to(grad, a.shape),
        casting=None,
    ),
    broadcast_to := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.broadcast_to),
        grad_a=lambda a, shape, grad: unbroadcast(grad=grad, target_shape=a.shape),
        casting=None,
    ),
    atleast_1d := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.atleast_1d),
        grad=lambda a, grad: grad,
        casting=None,
    ),
    atleast_2d := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.atleast_2d),
        grad=lambda a, grad: grad,
        casting=None,
    ),
    atleast_3d := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.atleast_3d),
        grad=lambda a, grad: grad,
        casting=None,
    ),
    copy := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.copy),
        grad_a=lambda a, grad: grad,
        casting=None,
    ),
    getitem := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(lambda a, key: a[key]),
        grad_a=getitem_grad,
        casting=None,
        op_name="index",
    ),
    clip := ops.generate_ternary_op_func(
        forward_func=ops.as_minidiff(np.clip),
        grad_a=lambda a, grad, a_min=None, a_max=None: grad
        * logical_and(
            a > float("-inf") if a_min is None else a_min,
            a < float("inf") if a_max is None else a_max,
        ),
        casting=None,
    ),
    reshape := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.reshape),
        grad_a=lambda a, b, grad: grad.reshape(a.shape),
        casting=None,
    ),
    matmul := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.matmul),
        grad_a=lambda a, b, grad: matmul(grad, b.t),
        grad_b=lambda a, b, grad: matmul(a.t, grad),
        tensor_only=True,
        casting=None,
    ),
    tensordot := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.tensordot),
        grad_a=tensordot_grad_a,
        grad_b=tensordot_grad_b,
        tensor_only=True,
        propagate_kwargs=True,
        casting=None,
    ),
    add := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.add),
        grad_a=lambda a, b, grad: grad,
        grad_b=lambda a, b, grad: grad,
    ),
    subtract := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.subtract),
        grad_a=lambda a, b, grad: grad,
        grad_b=lambda a, b, grad: -grad,
    ),
    multiply := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.multiply),
        grad_a=lambda a, b, grad: grad * b,
        grad_b=lambda a, b, grad: grad * a,
    ),
    true_divide := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.true_divide),
        grad_a=lambda a, b, grad: grad / b,
        grad_b=lambda a, b, grad: (-grad * a) / (b**2),
    ),
    floor_divide := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.floor_divide),
        is_differentiable=False,
    ),
    power := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.power),
        grad_a=lambda a, b, grad: grad * b * (a ** (b - 1)),
        grad_b=lambda a, b, grad: grad * log(a) * a**b,
    ),
    square := lambda a, **kwargs: power(a, 2, **kwargs),
    sqrt := lambda a, **kwargs: power(a, 0.5, **kwargs),
    floor := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.floor),
        is_differentiable=False,
    ),
    ceil := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.ceil),
        is_differentiable=False,
    ),
    cos := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.cos),
        grad=lambda a, grad: grad * -sin(a),
    ),
    sin := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.sin),
        grad=lambda a, grad: grad * cos(a),
    ),
    tan := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.tan),
        grad=lambda a, grad: grad * (1 / cos(a) ** 2),
    ),
    cosh := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.cosh),
        grad=lambda a, grad: grad * sinh(a),
    ),
    sinh := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.sinh),
        grad=lambda a, grad: grad * cosh(a),
    ),
    tanh := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.tanh),
        grad=lambda a, grad: grad * (1 / cosh(a) ** 2),
    ),
    exp := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.exp),
        grad=lambda a, grad: grad * exp(a),
    ),
    log := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.log),
        grad=lambda a, grad: grad / a,
    ),
    sum := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.sum),
        grad=lambda a, grad: grad,
        casting=None,
    ),
    mean := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.mean),
        grad=mean_grad,
        propagate_kwargs=True,
        casting=None,
    ),
    greater := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.greater),
        is_differentiable=False,
        casting=None,
    ),
    greater_equal := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.greater_equal),
        is_differentiable=False,
        casting=None,
    ),
    less := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.less),
        is_differentiable=False,
        casting=None,
    ),
    less_equal := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.less_equal),
        is_differentiable=False,
        casting=None,
    ),
    equal := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.equal),
        is_differentiable=False,
        casting=None,
    ),
    not_equal := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.not_equal),
        is_differentiable=False,
        casting=None,
    ),
    logical_and := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.logical_and),
        is_differentiable=False,
    ),
    logical_or := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.logical_or),
        is_differentiable=False,
    ),
    logical_not := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.logical_not),
        is_differentiable=False,
    ),
    logical_xor := ops.generate_binary_op_func(
        forward_func=ops.as_minidiff(np.logical_xor),
        is_differentiable=False,
    ),
    sign := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.sign),
        is_differentiable=False,
    ),
    absolute := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.absolute),
        grad=lambda a, grad: grad * sign(a),
    ),
    abs := absolute,
    all := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.all),
        is_differentiable=False,
        casting=None,
    ),
    any := ops.generate_unary_op_func(
        forward_func=ops.as_minidiff(np.any),
        is_differentiable=False,
        casting=None,
    ),
]


__all__ = get_exported_var_names(local_vars=dict(locals()), exported_vars=exported_ops)
