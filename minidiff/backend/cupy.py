from __future__ import annotations

from builtins import bool as py_bool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, Optional, Sequence, Tuple, Union

    import cupy.typing as cpt

import cupy as cp
from cupy.lib._shape_base import internal


# https://github.com/cupy/cupy/blob/main/cupy/lib/_shape_base.py#L156
def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over

    if not cp.issubdtype(indices.dtype, cp.integer):
        raise IndexError("`indices` must be an integer array")
    if len(arr_shape) != indices.ndim:
        raise ValueError("`indices` and `arr` must have the same number of dimensions")

    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal cupy.arange calls,
    # with the requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(cp.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


tensor_constructor = cp.array
tensor_class = cp.ndarray

# op functions
absolute = cp.absolute
all = cp.all
any = cp.any
argmax = cp.argmax
argmin = cp.argmin
argwhere = cp.argwhere
atleast_1d = cp.atleast_1d
atleast_2d = cp.atleast_2d
atleast_3d = cp.atleast_3d
ceil = cp.ceil
copy = cp.copy
cos = cp.cos
cosh = cp.cosh
exp = cp.exp


def flatten(a: cp.ndarray, order="C"):
    return a.flatten(order=order)


flip = cp.flip
floor = cp.floor
invert = cp.invert
log = cp.log
logical_not = cp.logical_not
max = cp.max
mean = cp.mean
min = cp.min
prod = cp.prod


def ravel(a: cp.ndarray, order="C"):
    return a.ravel(order=order)


sign = cp.sign
sin = cp.sin
sinh = cp.sinh
squeeze = cp.squeeze
std = cp.std
sum = cp.sum
tan = cp.tan
tanh = cp.tanh
transpose = cp.transpose
add = cp.add


def astype(x: cp.array, dtype: cp.dtype, **kwargs):
    return x.astype(dtype, **kwargs)


broadcast_to = cp.broadcast_to
dot = cp.dot
equal = cp.equal


def expand_dims(a: cp.array, axis: Union[int, Sequence[int]]) -> cp.array:
    return cp.expand_dims(a, tuple(axis))


# expand_dims = cp.expand_dims
floor_divide = cp.floor_divide


def getitem(a: cp.ndarray, key: Any):
    return a[key]


greater = cp.greater
greater_equal = cp.greater_equal
less = cp.less
less_equal = cp.less_equal
logical_and = cp.logical_and
logical_or = cp.logical_or
logical_xor = cp.logical_xor
matmul = cp.matmul
mod = cp.mod
multiply = cp.multiply
not_equal = cp.not_equal
power = cp.power
reshape = cp.reshape
subtract = cp.subtract
tensordot = cp.tensordot
true_divide = cp.true_divide
clip = cp.clip
swapaxes = cp.swapaxes
where = cp.where

# tensor functions
ones_like = cp.ones_like
ones = cp.ones
zeros_like = cp.zeros_like
zeros = cp.zeros
full_like = cp.full_like
full = cp.full
concatenate = cp.concatenate
index_add = cp.add.at
isin = cp.isin
unravel_index = cp.unravel_index
take_along_axis = cp.take_along_axis


# https://github.com/cupy/cupy/blob/main/cupy/lib/_shape_base.py#L182
def put_along_axis(
    arr: cp.ndarray, indices: cp.ndarray, values: cpt.ArrayLike, axis: Optional[int]
):
    if axis is None:
        if indices.ndim != 1:
            raise NotImplementedError("Tuple setitem isn't supported for flatiter.")
        # put is roughly equivalent to a.flat[ind] = values
        cp.put(arr, indices, values)
    else:
        axis = internal._normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

        # use the fancy index
        arr[_make_along_axis_idx(arr_shape, indices, axis)] = values


repeat = cp.repeat
tile = cp.tile
arange = cp.arange
stack = cp.stack
save = cp.save
load = cp.load
choice = cp.random.choice
rand = cp.random.rand
randint = cp.random.randint
randn = cp.random.randn
binomial = cp.random.binomial
permutation = cp.random.permutation
shuffle = cp.random.shuffle
split = cp.split


# tensor properties
def tensor_shape(data: cp.ndarray) -> Tuple[int, ...]:
    return data.shape


def tensor_size(data: cp.ndarray) -> int:
    return data.size


def tensor_ndim(data: cp.ndarray) -> int:
    return data.ndim


def tensor_dtype(data: cp.ndarray) -> dtype:
    return data.dtype


def tensor_item(data: cp.ndarray) -> Any:
    return data.item()


def repr(data: cp.ndarray) -> str:
    return data.__repr__()


def len(data: cp.ndarray) -> int:
    return data.__len__()


def array_interface(data: cp.ndarray) -> Dict[str, Any]:
    return data.__array_interface__


def array(
    data: cp.ndarray,
    dtype: Optional[cp.dtype] = None,
    copy: Optional[py_bool] = None,
) -> cp.ndarray:
    if dtype != data.dtype:
        if not copy:
            raise ValueError("attempted cast, but copies are not permitted")
        return data.astype(dtype=dtype)
    if copy:
        return data.copy()
    return data


# dtypes
dtype = cp.dtype
float64 = cp.float64
float32 = cp.float32
float16 = cp.float16
uint64 = cp.uint64
uint32 = cp.uint32
uint16 = cp.uint16
uint8 = cp.uint8
int64 = cp.int64
int32 = cp.int32
int16 = cp.int16
int8 = cp.int8
bool = cp.bool_

nan = cp.nan

as_numpy = cp.asnumpy
