from __future__ import annotations

from builtins import bool as py_bool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

import minidiff.backend as backend


class mlx_backend(backend.Backend):
    @staticmethod
    def tensor_constructor(*args, **kwargs) -> mx.array:
        return mx.stop_gradient(mx.array(*args, **kwargs))

    tensor_class = mx.array

    # op functions
    absolute = mx.abs
    all = mx.all
    any = mx.any
    argmax = mx.argmax
    argmin = mx.argmin

    @staticmethod
    def argwhere(a: mx.array) -> mx.array:
        flat_indices = mx.repeat(mx.arange(a.size)[:, None], a.ndim, axis=1)
        shape = mx.array(a.shape)
        divisor = a.size // mx.cumprod(shape)
        modulo = divisor * shape
        indices = (flat_indices % modulo) // divisor
        return mx.array([index for index, x in zip(indices, a.flatten()) if x])

    atleast_1d = mx.atleast_1d
    atleast_2d = mx.atleast_2d
    atleast_3d = mx.atleast_3d
    ceil = mx.ceil

    @staticmethod
    def copy(a: mx.array) -> mx.array:
        return mx.array(a)

    cos = mx.cos
    cosh = mx.cosh
    exp = mx.exp

    @staticmethod
    def flatten(a: mx.array, order="C"):
        return a.flatten()

    @staticmethod
    def flip(a: mx.array, axis: Optional[Union[int, Sequence[int]]]):
        if axis is None:
            slices = [slice(None, None, -1) for _ in range(a.ndim)]
        elif isinstance(axis, int):
            slices = [
                slice(None, None, -1) if i == axis else slice(None, None, None)
                for i in range(a.ndim)
            ]
        else:
            slices = [
                slice(None, None, -1) if i in axis else slice(None, None, None)
                for i in range(a.ndim)
            ]
        return a[*slices]

    floor = mx.floor
    invert = mx.bitwise_invert
    log = mx.log
    logical_not = mx.logical_not
    max = mx.max
    mean = mx.mean
    min = mx.min
    prod = mx.prod

    @staticmethod
    def ravel(a: mx.array, order="C"):
        return a.flatten()

    sign = mx.sign
    sin = mx.sin
    sinh = mx.sinh
    squeeze = mx.squeeze
    std = mx.std
    sum = mx.sum
    tan = mx.tan
    tanh = mx.tanh
    transpose = mx.transpose

    @staticmethod
    def add(*args, **kwargs):
        return mx.add(*args, **kwargs)

    @staticmethod
    def astype(a: mx.array, dtype: mx.Dtype) -> mx.array:
        return a.astype(dtype)

    broadcast_to = mx.broadcast_to

    @staticmethod
    def dot(a: mx.array, b: mx.array):
        return mx.matmul(a, b.T)

    equal = mx.equal
    expand_dims = mx.expand_dims
    floor_divide = mx.floor_divide

    @staticmethod
    def getitem(a: mx.array, key: Any):
        if isinstance(key, list):
            key = tuple(key)
        return a[key]

    greater = mx.greater
    greater_equal = mx.greater_equal
    less = mx.less
    less_equal = mx.less_equal
    logical_and = mx.logical_and
    logical_or = mx.logical_or
    logical_xor = mx.bitwise_xor
    matmul = mx.matmul
    mod = mx.remainder
    multiply = mx.multiply
    not_equal = mx.not_equal
    power = mx.power

    @staticmethod
    def reshape(*args, order="C", **kwargs) -> mx.array:
        return mx.reshape(*args, **kwargs)

    subtract = mx.subtract
    tensordot = mx.tensordot
    true_divide = mx.divide
    clip = mx.clip
    swapaxes = mx.swapaxes
    where = mx.where

    # tensor functions
    ones_like = mx.ones_like
    ones = mx.ones
    zeros_like = mx.zeros_like
    zeros = mx.zeros

    @staticmethod
    def full_like(a: mx.array, fill_value, dtype: mx.Dtype) -> mx.array:
        return mx.full(a.shape, fill_value, dtype=dtype)

    full = mx.full

    concatenate = mx.concatenate

    @staticmethod
    def index_add(a: mx.array, indices: mx.array, b: Optional[mx.array] = None):
        accumulated = mx.zeros_like(a)
        accumulated = accumulated.at[indices].add(b)
        a += accumulated

    @staticmethod
    def isin(element: mx.array, test_elements: mx.array, **kwargs) -> mx.array:
        return mx.array(np.isin(element, test_elements, **kwargs))

    @staticmethod
    def unravel_index(indices: mx.array, shape: Sequence[int]) -> mx.array:
        return mx.array(np.unravel_index(indices, shape))

    take_along_axis = mx.take_along_axis

    @staticmethod
    def vmap(fun: Callable) -> Callable:
        return mx.vmap(fun)

    @staticmethod
    def put_along_axis(
        array: mx.array, indices: mx.array, values: mx.array, axis: Optional[int]
    ) -> None:
        array[:] = mx.put_along_axis(array, indices, values, axis=axis)

    repeat = mx.repeat
    tile = mx.tile
    arange = mx.arange
    stack = mx.stack
    save = mx.save
    load = mx.load

    @staticmethod
    def choice(*args, **kwargs) -> mx.array:
        return mx.array(np.random.choice(*args, **kwargs))

    @staticmethod
    def rand(*dims) -> mx.array:
        return mx.random.uniform(shape=dims)

    @staticmethod
    def randint(
        low: Union[int, Sequence[int]],
        high: Optional[Union[int, Sequence[int]]] = None,
        size: Optional[Union[int, Sequence[int]]] = None,
    ) -> mx.array:
        if not isinstance(low, mx.array):
            low = mx.array(low)
        if not isinstance(high, mx.array):
            high = mx.array(high)
        if size is None:
            size = low.shape
        return mx.random.randint(low, high, shape=size)

    @staticmethod
    def randn(*dims: int) -> mx.array:
        return mx.random.normal(dims)

    @staticmethod
    def binomial(*args, **kwargs) -> mx.array:
        return mx.array(np.random.binomial(*args, **kwargs))

    permutation = mx.random.permutation

    @staticmethod
    def shuffle(a: mx.array, *args, **kwargs) -> mx.array:
        raise NotImplementedError("mlx has no random shuffle function")

    split = mx.split

    # tensor properties
    @staticmethod
    def tensor_shape(data: mx.array) -> Tuple[int, ...]:
        return data.shape

    @staticmethod
    def tensor_size(data: mx.array) -> int:
        return data.size

    @staticmethod
    def tensor_ndim(data: mx.array) -> int:
        return data.ndim

    @staticmethod
    def tensor_dtype(data: mx.array) -> mx.Dtype:
        return data.dtype

    @staticmethod
    def tensor_item(data: mx.array) -> Any:
        return data.item()

    @staticmethod
    def repr(data: mx.array) -> str:
        return data.__repr__()

    @staticmethod
    def len(data: mx.array) -> int:
        return data.__len__()

    @staticmethod
    def array_interface(data: mx.array) -> Dict[str, Any]:
        return data.__array_interface__

    @staticmethod
    def array(
        data: mx.array,
        dtype: Optional[mx.Dtype] = None,
        copy: Optional[py_bool] = None,
    ) -> mx.array:
        if dtype is not None and dtype != data.dtype:
            if not copy:
                raise ValueError("attempted cast, but copies are not permitted")
            return np.array(data, dtype=dtype)
        return np.array(data)

    # dtypes
    dtype = mx.Dtype
    float64 = mx.float64
    float32 = mx.float32
    float16 = mx.float16
    uint64 = mx.uint64
    uint32 = mx.uint32
    uint16 = mx.uint16
    uint8 = mx.uint8
    int64 = mx.int64
    int32 = mx.int32
    int16 = mx.int16
    int8 = mx.int8
    bool = mx.bool_

    nan = mx.nan

    @staticmethod
    def as_numpy(a: mx.array) -> np.array:
        return np.array(a)
