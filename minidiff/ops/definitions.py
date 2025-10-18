from __future__ import annotations

from builtins import min as py_min
from math import prod as py_prod
from typing import TYPE_CHECKING

import minidiff as md
import minidiff.backend as backend
import minidiff.ops.wrapping as wrapping

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Sequence, Tuple, Union

    import minidiff.typing as mdt


def squeeze_grad(
    a: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    if axis is None:
        axis = [i for i, dim in enumerate(a.shape) if dim == 1]
    return expand_dims(grad, axis)


def tensordot_grad_a(
    a: md.Tensor,
    b: md.Tensor,
    grad: md.Tensor,
    axes: Union[int, Sequence[Tuple[int, ...]]] = 2,
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
    axes: Union[int, Sequence[Tuple[int, ...]]] = 2,
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
    max_indices = argmax(a, axis=axis, keepdims=True)
    grad = grad.reshape(max_indices.shape)
    ret = md.zeros_like(a)
    md.put_along_axis(ret, max_indices, grad, axis=axis)
    return ret


def min_grad(
    a: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    min_indices = argmin(a, axis=axis, keepdims=True)
    grad = grad.reshape(min_indices.shape)
    ret = md.zeros_like(a)
    md.put_along_axis(ret, min_indices, grad, axis=axis)
    return ret


def prod_grad(
    a: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    if axis == ():
        return grad.reshape(a.shape)
    multiplied = prod(a, axis=axis, keepdims=True)
    grad = grad.reshape(multiplied.shape)
    # will be zero anyway over axes where an element is 0, just do this for numerical stability
    return md.where(a == 0, 0, grad * multiplied / a)


def transpose_grad(
    a: md.Tensor, grad: md.Tensor, axes: Optional[Union[int, Tuple[int]]] = None
) -> md.Tensor:
    if axes is None:
        return transpose(grad)
    backward_axes = [-1] * len(axes)
    for i, dim in enumerate(axes):
        backward_axes[dim.item()] = i
    return transpose(grad, axes=backward_axes)


# if broadcasting happened during the forward pass, you need to correctly
# sum up the correct dimensions so that the gradients match up
def unbroadcast_forward(a: md.Tensor, target_shape: Sequence[int]) -> md.Tensor:
    if a.shape == target_shape:
        return a
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
    ndims = py_min(len(target_shape), a.ndim)
    stretched_axes = tuple(
        i for i in range(ndims) if a.shape[i] > 1 and target_shape[i] == 1
    )
    if len(stretched_axes) != 0:
        a = a.sum(axis=stretched_axes, keepdims=True)

    # final reshape operation that can upsample if necessary
    if a.size == py_prod(target_shape):
        return a.reshape(target_shape)

    return broadcast_to(a, target_shape)


def getitem_grad(a: md.Tensor, key: Any, grad: md.Tensor) -> md.Tensor:
    ret = md.zeros_like(a)
    md.index_add(ret, key, grad)
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
    return grad / prod(multiplied_dims)


# -------------------- UNARY FUNCS --------------------
absolute: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.absolute),
    grad=lambda a, grad: grad * sign(a),
)
abs = absolute
all: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.all),
    is_differentiable=False,
)
any: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.any),
    is_differentiable=False,
)
argmax: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.argmax),
    is_differentiable=False,
)
argmin: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.argmin),
    is_differentiable=False,
)
argwhere: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.argwhere),
    is_differentiable=False,
)
atleast_1d: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.atleast_1d),
    grad=lambda a, grad: grad,
)
atleast_2d: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.atleast_2d),
    grad=lambda a, grad: grad,
)
atleast_3d: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.atleast_3d),
    grad=lambda a, grad: grad,
)
ceil: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.ceil),
    is_differentiable=False,
)
copy: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.copy),
    grad=lambda a, grad: grad,
)
cos: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.cos),
    grad=lambda a, grad: grad * -sin(a),
)
cosh: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.cosh),
    grad=lambda a, grad: grad * sinh(a),
)
exp: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.exp),
    grad=lambda a, grad: grad * exp(a),
)
flatten: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.flatten),
    grad=lambda a, grad, order="C": reshape(grad, a.shape, order=order),
)
flip: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.flip),
    grad=lambda a, grad, **kwargs: flip(grad, **kwargs),
    propagate_kwargs=True,
)
floor: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.floor),
    is_differentiable=False,
)
invert: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.invert),
    is_differentiable=False,
)
log: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.log),
    grad=lambda a, grad: grad / a,
)
logical_not: Callable[[mdt.TensorLike], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.logical_not),
    is_differentiable=False,
)
max: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.max),
    grad=max_grad,
    propagate_kwargs=True,
)
mean: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.mean),
    grad=mean_grad,
    propagate_kwargs=True,
)
min: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.min),
    grad=min_grad,
    propagate_kwargs=True,
)
prod: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.prod),
    grad=prod_grad,
    propagate_kwargs=True,
)
ravel: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.ravel),
    grad=lambda a, grad, order="C": reshape(grad, a.shape, order=order),
)
sign: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.sign),
    is_differentiable=False,
)
sin: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.sin),
    grad=lambda a, grad: grad * cos(a),
)
sinh: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.sinh),
    grad=lambda a, grad: grad * cosh(a),
)


def sqrt(a: md.Tensor, **kwargs) -> md.Tensor:
    return power(a, 0.5, **kwargs)


def square(a: md.Tensor, **kwargs) -> md.Tensor:
    return power(a, 2, **kwargs)


squeeze: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.squeeze),
    grad=squeeze_grad,
)
std: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.std),
    grad=lambda a, grad, axis=None, **kwargs: grad
    * (a - mean(a, axis=axis))
    / (std(a, axis=axis, **kwargs) * prod(a.shape[axis])),
)
sum: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.sum),
    grad=lambda a, grad: grad,
)
tan: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.tan),
    grad=lambda a, grad: grad * (1 / cos(a) ** 2),
)
tanh: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.tanh),
    grad=lambda a, grad: grad * (1 / cosh(a) ** 2),
)
transpose: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.transpose),
    grad=transpose_grad,
    propagate_kwargs=True,
)


# -------------------- BINARY FUNCS --------------------
add: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.add),
        grad_a=lambda a, b, grad: grad,
        grad_b=lambda a, b, grad: grad,
    )
)
astype: Callable[[md.Tensor, mdt.dtype], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.astype),
    grad_a=lambda a, dtype, grad: grad.astype(a.dtype),
)
broadcast_to: Callable[[md.Tensor, Sequence[int]], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.broadcast_to),
        grad_a=lambda a, shape, grad: unbroadcast(grad, a.shape),
    )
)
dot: Callable[[md.Tensor, md.Tensor], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.dot),
    grad_a=lambda a, b, grad: grad * b,
    grad_b=lambda a, b, grad: grad * a,
)
equal: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.equal),
        is_differentiable=False,
    )
)
expand_dims: Callable[[md.Tensor, Union[int, Sequence[int]]], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.expand_dims),
        grad_a=lambda a, axis, grad: squeeze(grad, axis=axis),
    )
)
floor_divide: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.floor_divide),
        is_differentiable=False,
    )
)
getitem: Callable[[md.Tensor, Any], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.getitem),
    grad_a=getitem_grad,
    op_name="index",
)
greater: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.greater),
        is_differentiable=False,
    )
)
greater_equal: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.greater_equal),
        is_differentiable=False,
    )
)
less: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.less),
        is_differentiable=False,
    )
)
less_equal: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.less_equal),
        is_differentiable=False,
    )
)
logical_and: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.logical_and),
        is_differentiable=False,
    )
)
logical_or: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.logical_or),
        is_differentiable=False,
    )
)
logical_xor: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.logical_xor),
        is_differentiable=False,
    )
)
matmul: Callable[[md.Tensor, md.Tensor], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.matmul),
    grad_a=lambda a, b, grad: matmul(grad, b.T),
    grad_b=lambda a, b, grad: matmul(a.T, grad),
    tensor_only=True,
)
mod: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.mod),
        grad_a=lambda a, b, grad: md.where(a % b == 0, 0, grad),
        grad_b=lambda a, b, grad: md.where(a % b == 0, 0, grad),
    )
)
multiply: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.multiply),
        grad_a=lambda a, b, grad: grad * b,
        grad_b=lambda a, b, grad: grad * a,
    )
)
not_equal: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.not_equal),
        is_differentiable=False,
    )
)
power: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.power),
        grad_a=lambda a, b, grad: grad * b * (a ** (b - 1)),
        grad_b=lambda a, b, grad: grad * log(a) * a**b,
    )
)
reshape: Callable[[md.Tensor, Union[int, Sequence[int]]], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.reshape),
        grad_a=lambda a, b, grad: grad.reshape(a.shape),
    )
)
subtract: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.subtract),
        grad_a=lambda a, b, grad: grad,
        grad_b=lambda a, b, grad: -grad,
    )
)
tensordot: Callable[[md.Tensor, md.Tensor], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.tensordot),
    grad_a=tensordot_grad_a,
    grad_b=tensordot_grad_b,
    tensor_only=True,
    propagate_kwargs=True,
)
true_divide: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(backend.true_divide),
        grad_a=lambda a, b, grad: grad / b,
        grad_b=lambda a, b, grad: grad * (-a / b**2),
    )
)
unbroadcast: Callable[[md.Tensor, Sequence[int]], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=unbroadcast_forward,
        grad_a=lambda a, shape, grad: broadcast_to(grad, a.shape),
    )
)
# -------------------- TERNARY FUNCS --------------------
clip: Callable[
    [md.Tensor, Optional[mdt.TensorLike], Optional[mdt.TensorLike]], md.Tensor
] = wrapping.create_ternary_op_func(
    forward_func=wrapping.as_minidiff(backend.clip),
    grad_a=lambda a, grad, a_min=None, a_max=None: grad
    * logical_and(
        1 if a_min is None else a > a_min,
        1 if a_max is None else a < a_max,
    ),
)
swapaxes: Callable[[md.Tensor, int, int], md.Tensor] = wrapping.create_ternary_op_func(
    forward_func=wrapping.as_minidiff(backend.swapaxes),
    grad_a=lambda a, axis1, axis2, grad, **kwargs: swapaxes(
        grad, axis1, axis2, **kwargs
    ),
    propagate_kwargs=True,
)
where: Callable[[md.Tensor, md.Tensor, md.Tensor], md.Tensor] = (
    wrapping.create_ternary_op_func(
        forward_func=wrapping.as_minidiff(backend.where),
        grad_b=lambda condition, b, c, grad: grad * condition,
        grad_c=lambda condition, b, c, grad: grad * (1 - condition),
    )
)

__all__ = [
    "absolute",
    "abs",
    "all",
    "any",
    "argmax",
    "argmin",
    "argwhere",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "ceil",
    "copy",
    "cos",
    "cosh",
    "exp",
    "flatten",
    "flip",
    "floor",
    "invert",
    "log",
    "logical_not",
    "max",
    "min",
    "mean",
    "prod",
    "ravel",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "squeeze",
    "std",
    "sum",
    "tan",
    "tanh",
    "transpose",
    "add",
    "astype",
    "broadcast_to",
    "dot",
    "equal",
    "expand_dims",
    "floor_divide",
    "getitem",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "matmul",
    "mod",
    "multiply",
    "not_equal",
    "power",
    "reshape",
    "subtract",
    "tensordot",
    "true_divide",
    "unbroadcast",
    "clip",
    "swapaxes",
    "where",
]
