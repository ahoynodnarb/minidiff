from __future__ import annotations

from builtins import min as py_min
from math import prod as py_prod
from typing import TYPE_CHECKING

import minidiff as md
import minidiff.backend as backend
import minidiff.ops.wrapping as wrapping

if TYPE_CHECKING:
    from typing import Any, Optional, Sequence, Tuple, Union


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


def std_grad(
    x: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    mu = mean(x, axis=axis)
    N = py_prod([dim for i, dim in enumerate(x.shape) if i in axis])
    return grad * (x - mu) / (std(x, axis=axis, **kwargs) * N)


def sum_grad(
    x: md.Tensor,
    grad: md.Tensor,
    axis: Optional[Union[int, Tuple[int]]] = None,
    **kwargs,
) -> md.Tensor:
    if isinstance(axis, int):
        axis = tuple(axis)
    if axis is None or not axis:
        return grad
    shape = x.shape
    ndim = len(x.shape)

    summed_indices = [i for i in range(ndim) if i in axis]
    summed_dims = [shape[i] for i in summed_indices]
    summed_ndim = len(summed_indices)

    n_compressed = ndim - summed_ndim

    tiled_axes = summed_dims + [1] * n_compressed
    prepended = md.tile(grad, tiled_axes)

    transposed_axes = [0] * ndim

    n_shifted = 0
    for i in reversed(range(ndim)):
        if n_shifted != summed_ndim and i == summed_indices[-(n_shifted + 1)]:
            transposed_axes[i] = summed_ndim - 1 - n_shifted
            n_shifted += 1
        else:
            transposed_axes[i] = i + n_shifted

    transposed = md.transpose(prepended, axes=transposed_axes)

    return transposed


# -------------------- UNARY FUNCS --------------------
absolute = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.absolute),
    grad=lambda x, grad: grad * sign(x),
)
abs = absolute
all = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.all),
    is_differentiable=False,
)
any = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.any),
    is_differentiable=False,
)
argmax = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.argmax),
    is_differentiable=False,
)
argmin = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.argmin),
    is_differentiable=False,
)
argwhere = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.argwhere),
    is_differentiable=False,
)
atleast_1d = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.atleast_1d),
    grad=lambda x, grad: grad,
)
atleast_2d = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.atleast_2d),
    grad=lambda x, grad: grad,
)
atleast_3d = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.atleast_3d),
    grad=lambda x, grad: grad,
)
ceil = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.ceil),
    is_differentiable=False,
)
copy = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.copy),
    grad=lambda x, grad: grad,
)
cos = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.cos),
    grad=lambda x, grad: grad * -sin(x),
)
cosh = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.cosh),
    grad=lambda x, grad: grad * sinh(x),
)
exp = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.exp),
    grad=lambda x, grad: grad * exp(x),
)
flatten = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.flatten),
    grad=lambda x, grad, order="C": reshape(grad, x.shape, order=order),
)
flip = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.flip),
    grad=lambda x, grad, **kwargs: flip(grad, **kwargs),
    propagate_kwargs=True,
)
floor = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.floor),
    is_differentiable=False,
)
invert = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.invert),
    is_differentiable=False,
)
log = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.log),
    grad=lambda x, grad: grad / x,
)
logical_not = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.logical_not),
    is_differentiable=False,
)
max = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.max),
    grad=max_grad,
    propagate_kwargs=True,
)
mean = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.mean),
    grad=mean_grad,
    propagate_kwargs=True,
)
min = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.min),
    grad=min_grad,
    propagate_kwargs=True,
)
prod = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.prod),
    grad=prod_grad,
    propagate_kwargs=True,
)
ravel = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.ravel),
    grad=lambda x, grad, order="C": reshape(grad, x.shape, order=order),
)
sign = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.sign),
    is_differentiable=False,
)
sin = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.sin),
    grad=lambda x, grad: grad * cos(x),
)
sinh = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.sinh),
    grad=lambda x, grad: grad * cosh(x),
)


def sqrt(a: md.Tensor, **kwargs) -> md.Tensor:
    return power(a, 0.5, **kwargs)


def square(a: md.Tensor, **kwargs) -> md.Tensor:
    return power(a, 2, **kwargs)


squeeze = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.squeeze),
    grad=squeeze_grad,
)
std = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.std),
    grad=std_grad,
    propagate_kwargs=True,
)
sum = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.sum),
    grad=sum_grad,
    propagate_kwargs=True,
)
tan = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.tan),
    grad=lambda x, grad: grad * (1 / cos(x) ** 2),
)
tanh = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.tanh),
    grad=lambda x, grad: grad * (1 / cosh(x) ** 2),
)
transpose = wrapping.create_unary_op_func(
    forward_func=wrapping.as_minidiff(backend.transpose),
    grad=transpose_grad,
    propagate_kwargs=True,
)


# -------------------- BINARY FUNCS --------------------
add = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.add),
    grad_x=lambda x, y, grad: grad,
    grad_y=lambda x, y, grad: grad,
)
astype = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.astype),
    grad_x=lambda x, dtype, grad: grad.astype(x.dtype),
)
broadcast_to = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.broadcast_to),
    grad_x=lambda x, shape, grad: unbroadcast(grad, x.shape),
)
dot = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.dot),
    grad_x=lambda x, y, grad: grad * y,
    grad_y=lambda x, y, grad: grad * x,
)
equal = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.equal),
    is_differentiable=False,
)
expand_dims = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.expand_dims),
    grad_x=lambda x, axis, grad: squeeze(grad, axis=axis),
)
floor_divide = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.floor_divide),
    is_differentiable=False,
)
getitem = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.getitem),
    grad_x=getitem_grad,
    op_name="index",
)
greater = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.greater),
    is_differentiable=False,
)
greater_equal = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.greater_equal),
    is_differentiable=False,
)
less = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.less),
    is_differentiable=False,
)
less_equal = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.less_equal),
    is_differentiable=False,
)
logical_and = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.logical_and),
    is_differentiable=False,
)
logical_or = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.logical_or),
    is_differentiable=False,
)
logical_xor = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.logical_xor),
    is_differentiable=False,
)
matmul = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.matmul),
    grad_x=lambda x, y, grad: matmul(grad, y.T),
    grad_y=lambda x, y, grad: matmul(x.T, grad),
    tensor_only=True,
)
mod = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.mod),
    grad_x=lambda x, y, grad: md.where(x % y == 0, 0, grad),
    grad_y=lambda x, y, grad: md.where(x % y == 0, 0, grad),
)
multiply = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.multiply),
    grad_x=lambda x, y, grad: grad * y,
    grad_y=lambda x, y, grad: grad * x,
)
not_equal = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.not_equal),
    is_differentiable=False,
)
power = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.power),
    grad_x=lambda x, y, grad: grad * y * (x ** (y - 1)),
    grad_y=lambda x, y, grad: grad * log(x) * x**y,
)
reshape = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.reshape),
    grad_x=lambda x, y, grad: grad.reshape(x.shape),
)
subtract = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.subtract),
    grad_x=lambda x, y, grad: grad,
    grad_y=lambda x, y, grad: -grad,
)
tensordot = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.tensordot),
    grad_x=tensordot_grad_x,
    grad_y=tensordot_grad_y,
    tensor_only=True,
    propagate_kwargs=True,
)
true_divide = wrapping.create_binary_op_func(
    forward_func=wrapping.as_minidiff(backend.true_divide),
    grad_x=lambda x, y, grad: grad / y,
    grad_y=lambda x, y, grad: grad * (-x / y**2),
)
unbroadcast = wrapping.create_binary_op_func(
    forward_func=unbroadcast_forward,
    grad_x=lambda x, shape, grad: broadcast_to(grad, x.shape),
)
# -------------------- TERNARY FUNCS --------------------
clip = wrapping.create_ternary_op_func(
    forward_func=wrapping.as_minidiff(backend.clip),
    grad_x=lambda x, a_min, a_max, grad: (
        grad
        * logical_and(
            1 if a_min is None else x > a_min,
            1 if a_max is None else x < a_max,
        )
    ),
)
swapaxes = wrapping.create_ternary_op_func(
    forward_func=wrapping.as_minidiff(backend.swapaxes),
    grad_x=lambda x, axis1, axis2, grad, **kwargs: swapaxes(
        grad, axis1, axis2, **kwargs
    ),
    propagate_kwargs=True,
)
where = wrapping.create_ternary_op_func(
    forward_func=wrapping.as_minidiff(backend.where),
    grad_y=lambda condition, y, z, grad: grad * condition,
    grad_z=lambda condition, y, z, grad: grad * (1 - condition),
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
