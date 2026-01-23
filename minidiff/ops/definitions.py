from __future__ import annotations

from builtins import min as py_min
from math import prod as py_prod
from typing import TYPE_CHECKING

import minidiff as md
import minidiff.ops.wrapping as wrapping
from minidiff.backend import current_backend

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


def tensordot_grad_x(
    x: md.Tensor,
    y: md.Tensor,
    grad: md.Tensor,
    axes: Union[int, Sequence[Tuple[int, ...]]] = 2,
) -> md.Tensor:
    if isinstance(axes, int):
        axes_x = tuple(range(x.ndim - axes, x.ndim))
        axes_y = tuple(range(axes))
        axes = (axes_x, axes_y)
    # indices of all dims in b not originally contracted in the forward tensordot
    uncontracted_x = tuple(i for i in range(x.ndim) if i not in axes[0])
    uncontracted_y = tuple(i for i in range(y.ndim) if i not in axes[1])
    # indices of all dims in grad that align with uncontracted_y
    grad_aligned = tuple(range(grad.ndim - len(uncontracted_y), grad.ndim))
    new_axes = (grad_aligned, uncontracted_y)
    result = tensordot(grad, y, axes=new_axes)
    # first few indices will be uncontracted in a, last few will be contracted in a (original forward pass)
    # need to transpose such that the first few take up the uncontracted a indices, and the last few take up the contracted a indices
    permutation_indices = [0] * x.ndim
    n_uncontracted_x = len(uncontracted_x)
    uncontracted_idx = 0
    contracted_idx = 0
    for i in range(x.ndim):
        if i < n_uncontracted_x:
            permutation_indices[uncontracted_x[uncontracted_idx]] = i
            uncontracted_idx += 1
        else:
            permutation_indices[axes[0][contracted_idx]] = i
            contracted_idx += 1

    reshaped = md.transpose(result, axes=permutation_indices)
    return reshaped


def tensordot_grad_y(
    x: md.Tensor,
    y: md.Tensor,
    grad: md.Tensor,
    axes: Union[int, Sequence[Tuple[int, ...]]] = 2,
) -> md.Tensor:
    if isinstance(axes, int):
        axes_x = tuple(range(x.ndim - axes, x.ndim))
        axes_y = tuple(range(axes))
        axes = (axes_x, axes_y)
    # indices of all dims in a not originally contracted in the forward tensordot
    uncontracted_x = tuple(i for i in range(x.ndim) if i not in axes[0])
    uncontracted_y = tuple(i for i in range(y.ndim) if i not in axes[1])
    # indices of all dims in grad that align with uncontracted_x
    grad_aligned = tuple(range(len(uncontracted_x)))
    new_axes = (uncontracted_x, grad_aligned)
    result = tensordot(x, grad, axes=new_axes)
    # first few indices of result will be contracted in a, last few will be uncontracted in b (original forward pass
    # need to transpose so that the last few take up the uncontracted b indices, and the first few take up the original contracted indices
    n_contracted_x = len(axes[0])
    contracted_idx = 0
    uncontracted_idx = 0
    permutation_indices = [0] * y.ndim
    for i in range(y.ndim):
        if i < n_contracted_x:
            permutation_indices[axes[1][contracted_idx]] = i
            contracted_idx += 1
        else:
            permutation_indices[uncontracted_y[uncontracted_idx]] = i
            uncontracted_idx += 1

    reshaped = md.transpose(result, axes=permutation_indices)
    return reshaped


def max_grad(
    x: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    max_indices = argmax(x, axis=axis, keepdims=True)
    grad = grad.reshape(max_indices.shape)
    ret = md.zeros_like(x)
    md.put_along_axis(ret, max_indices, grad, axis=axis)
    return ret


def min_grad(
    x: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    min_indices = argmin(x, axis=axis, keepdims=True)
    grad = grad.reshape(min_indices.shape)
    ret = md.zeros_like(x)
    md.put_along_axis(ret, min_indices, grad, axis=axis)
    return ret


def prod_grad(
    x: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    if axis == ():
        return grad.reshape(x.shape)
    multiplied = prod(x, axis=axis, keepdims=True)
    grad = grad.reshape(multiplied.shape)
    # will be zero anyway over axes where an element is 0, just do this for numerical stability
    return md.where(x == 0, 0, grad * multiplied / x)


def transpose_grad(
    x: md.Tensor, grad: md.Tensor, axes: Optional[Union[int, Tuple[int]]] = None
) -> md.Tensor:
    if axes is None:
        return transpose(grad)
    backward_axes = [-1] * len(axes)
    for i, dim in enumerate(axes):
        backward_axes[dim.item()] = i
    return transpose(grad, axes=backward_axes)


# if broadcasting happened during the forward pass, you need to correctly
# sum up the correct dimensions so that the gradients match up
def unbroadcast_forward(x: md.Tensor, target_shape: Sequence[int]) -> md.Tensor:
    if x.shape == target_shape:
        return x
    # this collects the prepended axes
    # numpy inserts dimensions from the left when broadcasting
    # this just sums across those prepended dimensions
    len_prepended = x.ndim - len(target_shape)
    broadcasted_axes = tuple(range(len_prepended))
    if len(broadcasted_axes) != 0:
        x = x.sum(axis=broadcasted_axes)

    # this collects the axes that were stretched to span a greater dim
    # numpy will "stretch" dimensions so that dimensions of size 1 are tiled
    # so that they match the greater dimension.
    # we can undo this by just summing across those dimensions until we reach size 1 again
    ndims = py_min(len(target_shape), x.ndim)
    stretched_axes = tuple(
        i for i in range(ndims) if x.shape[i] > 1 and target_shape[i] == 1
    )
    if len(stretched_axes) != 0:
        x = x.sum(axis=stretched_axes, keepdims=True)

    # final reshape operation that can upsample if necessary
    if x.size == py_prod(target_shape):
        return x.reshape(target_shape)

    return broadcast_to(x, target_shape)


def getitem_grad(x: md.Tensor, key: Any, grad: md.Tensor) -> md.Tensor:
    ret = md.zeros_like(x)
    md.index_add(ret, key, grad)
    return ret


def mean_grad(
    x: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    if axis is None:
        return grad / x.size
    if axis == ():
        return grad
    if isinstance(axis, int):
        return grad / x.shape[axis]
    in_shape = x.shape
    multiplied_dims = md.Tensor([in_shape[dim] for dim in axis])
    return grad / prod(multiplied_dims)


def std_grad(x, grad, axis=None, **kwargs):
    mu = mean(x, axis=axis)
    N = py_prod([dim for i, dim in enumerate(x.shape) if i in axis])
    return grad * (x - mu) / (std(x, axis=axis, **kwargs) * N)


# -------------------- UNARY FUNCS --------------------
absolute: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.absolute),
    grad=lambda x, grad: grad * sign(x),
)
abs = absolute
all: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.all),
    is_differentiable=False,
)
any: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.any),
    is_differentiable=False,
)
argmax: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.argmax),
    is_differentiable=False,
)
argmin: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.argmin),
    is_differentiable=False,
)
argwhere: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.argwhere),
    is_differentiable=False,
)
atleast_1d: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.atleast_1d),
    grad=lambda x, grad: grad,
)
atleast_2d: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.atleast_2d),
    grad=lambda x, grad: grad,
)
atleast_3d: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.atleast_3d),
    grad=lambda x, grad: grad,
)
ceil: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.ceil),
    is_differentiable=False,
)
copy: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.copy),
    grad=lambda x, grad: grad,
)
cos: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.cos),
    grad=lambda x, grad: grad * -sin(x),
)
cosh: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.cosh),
    grad=lambda x, grad: grad * sinh(x),
)
exp: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.exp),
    grad=lambda x, grad: grad * exp(x),
)
flatten: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.flatten),
    grad=lambda x, grad, order="C": reshape(grad, x.shape, order=order),
)
flip: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.flip),
    grad=lambda x, grad, **kwargs: flip(grad, **kwargs),
    propagate_kwargs=True,
)
floor: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.floor),
    is_differentiable=False,
)
invert: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.invert),
    is_differentiable=False,
)
log: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.log),
    grad=lambda x, grad: grad / x,
)
logical_not: Callable[[mdt.TensorLike], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.logical_not),
    is_differentiable=False,
)
max: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.max),
    grad=max_grad,
    propagate_kwargs=True,
)
mean: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.mean),
    grad=mean_grad,
    propagate_kwargs=True,
)
min: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.min),
    grad=min_grad,
    propagate_kwargs=True,
)
prod: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.prod),
    grad=prod_grad,
    propagate_kwargs=True,
)
ravel: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.ravel),
    grad=lambda x, grad, order="C": reshape(grad, x.shape, order=order),
)
sign: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.sign),
    is_differentiable=False,
)
sin: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.sin),
    grad=lambda x, grad: grad * cos(x),
)
sinh: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.sinh),
    grad=lambda x, grad: grad * cosh(x),
)


def sqrt(a: md.Tensor, **kwargs) -> md.Tensor:
    return power(a, 0.5, **kwargs)


def square(a: md.Tensor, **kwargs) -> md.Tensor:
    return power(a, 2, **kwargs)


squeeze: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.squeeze),
    grad=squeeze_grad,
)
std: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.std),
    grad=std_grad,
    propagate_kwargs=True,
)
sum: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.sum),
    grad=lambda x, grad: grad,
)
tan: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.tan),
    grad=lambda x, grad: grad * (1 / cos(x) ** 2),
)
tanh: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.tanh),
    grad=lambda x, grad: grad * (1 / cosh(x) ** 2),
)
transpose: Callable[[md.Tensor], md.Tensor] = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.transpose),
    grad=transpose_grad,
    propagate_kwargs=True,
)


# -------------------- BINARY FUNCS --------------------
add: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.add),
        grad_x=lambda x, y, grad: grad,
        grad_y=lambda x, y, grad: grad,
    )
)
astype: Callable[[md.Tensor, mdt.dtype], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.astype),
    grad_x=lambda x, dtype, grad: grad.astype(x.dtype),
)
broadcast_to: Callable[[md.Tensor, Sequence[int]], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.broadcast_to),
        grad_x=lambda x, shape, grad: unbroadcast(grad, x.shape),
    )
)
dot: Callable[[md.Tensor, md.Tensor], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.dot),
    grad_x=lambda x, y, grad: grad * y,
    grad_y=lambda x, y, grad: grad * x,
)
equal: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.equal),
        is_differentiable=False,
    )
)
expand_dims: Callable[[md.Tensor, Union[int, Sequence[int]]], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.expand_dims),
        grad_x=lambda x, axis, grad: squeeze(grad, axis=axis),
    )
)
floor_divide: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.floor_divide),
        is_differentiable=False,
    )
)
getitem: Callable[[md.Tensor, Any], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.getitem),
    grad_x=getitem_grad,
    op_name="index",
)
greater: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.greater),
        is_differentiable=False,
    )
)
greater_equal: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.greater_equal),
        is_differentiable=False,
    )
)
less: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.less),
        is_differentiable=False,
    )
)
less_equal: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.less_equal),
        is_differentiable=False,
    )
)
logical_and: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.logical_and),
        is_differentiable=False,
    )
)
logical_or: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.logical_or),
        is_differentiable=False,
    )
)
logical_xor: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.logical_xor),
        is_differentiable=False,
    )
)
matmul: Callable[[md.Tensor, md.Tensor], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.matmul),
    grad_x=lambda x, y, grad: matmul(grad, y.T),
    grad_y=lambda x, y, grad: matmul(x.T, grad),
    tensor_only=True,
)
mod: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.mod),
        grad_x=lambda x, y, grad: md.where(x % y == 0, 0, grad),
        grad_y=lambda x, y, grad: md.where(x % y == 0, 0, grad),
    )
)
multiply: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.multiply),
        grad_x=lambda x, y, grad: grad * y,
        grad_y=lambda x, y, grad: grad * x,
    )
)
not_equal: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.not_equal),
        is_differentiable=False,
    )
)
power: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.power),
        grad_x=lambda x, y, grad: grad * y * (x ** (y - 1)),
        grad_y=lambda x, y, grad: grad * log(x) * x**y,
    )
)
reshape: Callable[[md.Tensor, Union[int, Sequence[int]]], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.reshape),
        grad_x=lambda x, y, grad: grad.reshape(x.shape),
    )
)
subtract: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.subtract),
        grad_x=lambda x, y, grad: grad,
        grad_y=lambda x, y, grad: -grad,
    )
)
tensordot: Callable[[md.Tensor, md.Tensor], md.Tensor] = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.tensordot),
    grad_x=tensordot_grad_x,
    grad_y=tensordot_grad_y,
    tensor_only=True,
    propagate_kwargs=True,
)
true_divide: Callable[[mdt.TensorLike, mdt.TensorLike], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.true_divide),
        grad_x=lambda x, y, grad: grad / y,
        grad_y=lambda x, y, grad: grad * (-x / y**2),
    )
)
unbroadcast: Callable[[md.Tensor, Sequence[int]], md.Tensor] = (
    wrapping.create_binary_op_func(
        forward_func=unbroadcast_forward,
        grad_x=lambda x, shape, grad: broadcast_to(grad, x.shape),
    )
)
# -------------------- TERNARY FUNCS --------------------
clip: Callable[
    [md.Tensor, Optional[mdt.TensorLike], Optional[mdt.TensorLike]], md.Tensor
] = wrapping.create_ternary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.clip),
    grad_x=lambda x, grad, a_min=None, a_max=None: grad
    * logical_and(
        1 if a_min is None else x > a_min,
        1 if a_max is None else x < a_max,
    ),
)
swapaxes: Callable[[md.Tensor, int, int], md.Tensor] = wrapping.create_ternary_op_func(
    forward_func=wrapping.as_minidiff(current_backend.swapaxes),
    grad_x=lambda x, axis1, axis2, grad, **kwargs: swapaxes(
        grad, axis1, axis2, **kwargs
    ),
    propagate_kwargs=True,
)
where: Callable[[md.Tensor, md.Tensor, md.Tensor], md.Tensor] = (
    wrapping.create_ternary_op_func(
        forward_func=wrapping.as_minidiff(current_backend.where),
        grad_y=lambda condition, y, z, grad: grad * condition,
        grad_z=lambda condition, y, z, grad: grad * (1 - condition),
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
