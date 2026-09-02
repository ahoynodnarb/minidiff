from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from builtins import bool as py_bool
    from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

    import numpy as np

    import minidiff.typing as mdt

cupy = "minidiff.backend.default_cupy"
mlx = "minidiff.backend.default_mlx"
numpy = "minidiff.backend.default_numpy"

_DEFAULT_BACKENDS = [cupy, mlx, numpy]

current_backend: Backend = None


class minidiff_backend_proxy:
    _instance: Backend = None

    def __getattr__(self, name):
        return getattr(self._instance, name)

    def __repr__(self):
        return self._instance.name


def _update_backend(new_backend: Backend):
    if not isinstance(new_backend, Backend):
        raise ValueError(f"{new_backend} is not of type {Backend}")

    global current_backend

    if current_backend is None:
        current_backend = minidiff_backend_proxy()

    # we have to update the existing current_backend so that all instances of current_backend
    # including those copied from `from minidiff.backend import current_backend` are updated
    current_backend._instance = new_backend


def _get_module_backend(module: Union[ModuleType, str]) -> Optional[Backend]:
    if isinstance(module, ModuleType):
        module_dict = module.__dict__
    else:
        module_dict = importlib.import_module(module).__dict__

    for obj in module_dict.values():
        if isinstance(obj, Backend):
            return obj

    return None


def set_backend(backend: Union[Backend, ModuleType, str], silent=False):
    if not isinstance(backend, Backend):
        possible_backend = _get_module_backend(backend)

        backend = possible_backend

    _update_backend(backend)

    if not silent:
        print(f"Using {backend} as backend")


def _attempt_backend_import():
    if current_backend is not None:
        return

    for module_name in _DEFAULT_BACKENDS:
        try:
            set_backend(module_name, silent=True)
            return
        except:
            continue

    raise Exception("could not find a suitable backend")


@runtime_checkable
class Backend(Protocol):
    tensor_class: mdt.BackendTensor

    def tensor_constructor(*args, **kwargs) -> Backend.mdt.BackendTensor: ...

    def absolute(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def abs(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def all(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def any(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def argmax(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def argmin(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def argwhere(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def atleast_1d(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def atleast_2d(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def atleast_3d(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def ceil(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def copy(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def cos(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def cosh(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def exp(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def flatten(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def flip(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def floor(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def invert(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def log(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def logical_not(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def max(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def mean(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def min(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def prod(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def ravel(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def sign(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def sin(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def sinh(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def sqrt(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def square(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def squeeze(
        x: mdt.BackendTensor, axis: Optional[Union[int, Sequence[int]]] = None
    ) -> mdt.BackendTensor: ...

    def std(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def sum(
        x: mdt.BackendTensor,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def tan(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def tanh(x: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def transpose(
        x: mdt.BackendTensor, axes: Optional[Sequence[int]] = None
    ) -> mdt.BackendTensor: ...

    def add(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def astype(x: mdt.BackendTensor, type: mdt.dtype) -> mdt.BackendTensor: ...

    def broadcast_to(
        x: mdt.BackendTensor, shape: Sequence[int]
    ) -> mdt.BackendTensor: ...

    def dot(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def equal(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def expand_dims(
        x: mdt.BackendTensor, axis: Union[int, Sequence[int]]
    ) -> mdt.BackendTensor: ...

    def floor_divide(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def getitem(x: mdt.BackendTensor, index: Any) -> mdt.BackendTensor: ...

    def greater(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def greater_equal(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def less(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def less_equal(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def logical_and(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def logical_or(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def logical_xor(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def matmul(x: mdt.BackendTensor, y: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def mod(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def multiply(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def not_equal(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def power(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def reshape(
        x: mdt.BackendTensor, shape: Union[int, Sequence[int]]
    ) -> mdt.BackendTensor: ...

    def subtract(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def tensordot(x: mdt.BackendTensor, y: mdt.BackendTensor) -> mdt.BackendTensor: ...

    def true_divide(
        x: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def unbroadcast(
        x: mdt.BackendTensor, shape: Sequence[int]
    ) -> mdt.BackendTensor: ...

    def clip(
        x: mdt.BackendTensor,
        a_min: Optional[Union[int, float, mdt.BackendTensor]],
        a_max: Optional[Union[int, float, mdt.BackendTensor]],
    ) -> mdt.BackendTensor: ...

    def swapaxes(x: mdt.BackendTensor, axis1: int, axis2: int) -> mdt.BackendTensor: ...

    def where(
        condition: Union[int, float, mdt.BackendTensor],
        y: Union[int, float, mdt.BackendTensor],
        z: Union[int, float, mdt.BackendTensor],
    ) -> mdt.BackendTensor: ...

    def ones_like(
        a: Union[int, float, mdt.BackendTensor], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def ones(
        shape: Union[int, Sequence[int]], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def zeros_like(
        a: Union[int, float, mdt.BackendTensor], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def zeros(
        shape: Union[int, Sequence[int]], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def full_like(
        a: mdt.BackendTensor,
        x: Union[int, float, mdt.BackendTensor],
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def full(
        shape: Union[int, Sequence[int]], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def concatenate(
        arrays: Sequence[Union[int, float, mdt.BackendTensor]],
        axis: Optional[int] = 0,
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def index_add(
        a: Union[int, float, mdt.BackendTensor],
        indices: Union[int, float, mdt.BackendTensor],
        b: Optional[Union[int, float, mdt.BackendTensor]] = None,
    ): ...

    def isin(
        element: Union[int, float, mdt.BackendTensor],
        test_elements: List[Union[int, float, mdt.BackendTensor]],
    ) -> mdt.BackendTensor: ...

    def unravel_index(
        indices: Union[int, float, mdt.BackendTensor],
        shape: Sequence[int],
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def vmap(
        fun: Callable[[mdt.BackendTensor], mdt.BackendTensor],
    ) -> Callable[[mdt.BackendTensor], mdt.BackendTensor]: ...

    def take_along_axis(
        arr: mdt.BackendTensor,
        indices: mdt.BackendTensor,
        axis: Optional[int] = None,
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def put_along_axis(
        arr: mdt.BackendTensor,
        indices: mdt.BackendTensor,
        values: Union[int, float, mdt.BackendTensor],
        axis: Optional[int],
    ): ...

    def repeat(
        a: Union[int, float, mdt.BackendTensor],
        repeats: Union[int, Sequence[int]],
        allow_grad: py_bool = False,
        axis: Optional[int] = None,
    ) -> mdt.BackendTensor: ...

    def tile(
        A: Union[int, float, mdt.BackendTensor],
        reps: Union[int, float, mdt.BackendTensor],
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def arange(
        *args: Union[int, float], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def stack(
        arrays: Sequence[mdt.BackendTensor],
        axis: Optional[int] = 0,
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def save(file, arr: Union[int, float, mdt.BackendTensor]): ...

    def load(file, allow_grad: py_bool = False) -> mdt.BackendTensor: ...

    def choice(
        a: Union[int, Union[int, float, mdt.BackendTensor]],
        size: Optional[Union[int, Sequence[int]]] = None,
        replace: py_bool = True,
        p: Optional[Union[int, float, mdt.BackendTensor]] = None,
    ) -> mdt.BackendTensor: ...

    def rand(
        *dims: Optional[int], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def randint(
        low: Union[int, Union[int, float, mdt.BackendTensor]],
        high: Optional[Union[int, Union[int, float, mdt.BackendTensor]]] = None,
        size: Optional[Union[int, Sequence[int]]] = None,
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def randn(
        *dims: Optional[int], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def binomial(
        n: Union[int, mdt.BackendTensor[int]],
        p: Union[float, mdt.BackendTensor[float]],
        size: Optional[Tuple[int]] = None,
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def permutation(
        x: Union[int, mdt.BackendTensor], allow_grad: py_bool = False
    ) -> mdt.BackendTensor: ...

    def shuffle(x: mdt.BackendTensor): ...

    def split(
        ary: mdt.BackendTensor,
        indices_or_sections: Union[int, Sequence[int]],
        axis: int = 0,
        allow_grad: py_bool = False,
    ) -> mdt.BackendTensor: ...

    def tensor_shape(data: mdt.BackendTensor) -> Tuple[int, ...]: ...

    def tensor_size(data: mdt.BackendTensor) -> int: ...

    def tensor_ndim(data: mdt.BackendTensor) -> int: ...

    def tensor_dtype(data: mdt.BackendTensor) -> mdt.dtype: ...

    def tensor_item(data: mdt.BackendTensor) -> Any: ...

    def repr(data: mdt.BackendTensor) -> str: ...

    def len(data: mdt.BackendTensor) -> int: ...

    def array_interface(data: mdt.BackendTensor) -> dict[str, Any]: ...

    def array(
        data: mdt.BackendTensor,
        dtype: Optional[mdt.dtype] = None,
        copy: Optional[py_bool] = None,
    ) -> mdt.BackendTensor: ...

    dtype: mdt.dtype

    float64: mdt.dtype

    float32: mdt.dtype

    float16: mdt.dtype

    uint64: mdt.dtype

    uint32: mdt.dtype

    uint16: mdt.dtype

    uint8: mdt.dtype

    int64: mdt.dtype

    int32: mdt.dtype

    int16: mdt.dtype

    int8: mdt.dtype

    bool: mdt.dtype

    nan: Any

    def as_numpy(a: mdt.BackendTensor) -> np.array: ...

    name: str
