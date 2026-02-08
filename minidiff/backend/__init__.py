from __future__ import annotations

import importlib
from argparse import ArgumentParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from builtins import bool as py_bool
    from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

    import numpy as np

_parser = ArgumentParser()
_parser.add_argument(
    "--backend", help="specify selected backend", required=False, default=None
)
_args = vars(_parser.parse_args())

_SPECIFIED_BACKEND = _args["backend"]
_DEFAULT_BACKENDS = [
    "minidiff.backend.cupy",
    "minidiff.backend.mlx",
    "minidiff.backend.numpy",
]


def import_backend(backend_name: str, package_name: Optional[str] = None) -> dict:
    # https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
    module = importlib.import_module(backend_name, package=package_name)
    module_dict = module.__dict__
    return module_dict


def attempt_import(possible_backend: Optional[str]) -> Optional[dict]:
    if possible_backend is None:
        return None
    try:
        return import_backend(possible_backend)
    except:
        return None


def attempt_backend_import():
    current_backend = None

    used_backend = None
    backend_exports = None

    backend_exports = attempt_import(_SPECIFIED_BACKEND)
    if backend_exports is not None:
        used_backend = _SPECIFIED_BACKEND
    else:
        for possible_backend in _DEFAULT_BACKENDS:
            backend_exports = attempt_import(possible_backend)
            if backend_exports is not None:
                used_backend = possible_backend
                break
        else:
            raise Exception("could not find a suitable backend")

    for export in backend_exports.values():
        if (
            isinstance(export, type)
            and export is not type(Backend)
            and issubclass(export, Backend)
        ):
            if _SPECIFIED_BACKEND is not None and _SPECIFIED_BACKEND != used_backend:
                print(
                    f"could not find backend named {_SPECIFIED_BACKEND}, defaulting to {used_backend} instead"
                )
            current_backend = export
            break

    if current_backend is None or used_backend is None:
        raise Exception("could not find a suitable backend")

    import_backend_funcs(current_backend)


def import_backend_funcs(current_backend: Backend):
    class_dict = current_backend.__dict__
    module_exports = [x for x in class_dict if not x.startswith("_")]
    pairs = {k: getattr(current_backend, k) for k in module_exports}

    globals().update(pairs)


class tensor_class:
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


def tensor_constructor(*args, **kwargs) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def absolute(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def abs(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def all(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def any(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def argmax(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def argmin(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def argwhere(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def atleast_1d(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def atleast_2d(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def atleast_3d(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def ceil(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def copy(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def cos(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def cosh(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def exp(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def flatten(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def flip(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def floor(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def invert(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def log(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def logical_not(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def max(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def mean(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def min(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def prod(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def ravel(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def sign(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def sin(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def sinh(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def sqrt(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def square(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def squeeze(
    x: tensor_class, axis: Optional[Union[int, Sequence[int]]] = None
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def std(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def sum(
    x: tensor_class,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tan(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tanh(x: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def transpose(x: tensor_class, axes: Optional[Sequence[int]] = None) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def add(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def astype(x: tensor_class, type: dtype) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def broadcast_to(x: tensor_class, shape: Sequence[int]) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def dot(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def equal(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def expand_dims(x: tensor_class, axis: Union[int, Sequence[int]]) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def floor_divide(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def getitem(x: tensor_class, index: Any) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def greater(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def greater_equal(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def less(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def less_equal(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def logical_and(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def logical_or(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def logical_xor(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def matmul(x: tensor_class, y: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def mod(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def multiply(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def not_equal(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def power(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def reshape(x: tensor_class, shape: Union[int, Sequence[int]]) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def subtract(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tensordot(x: tensor_class, y: tensor_class) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def true_divide(
    x: Union[int, float, tensor_class], y: Union[int, float, tensor_class]
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def unbroadcast(x: tensor_class, shape: Sequence[int]) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def clip(
    x: tensor_class,
    a_min: Optional[Union[int, float, tensor_class]],
    a_max: Optional[Union[int, float, tensor_class]],
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def swapaxes(x: tensor_class, axis1: int, axis2: int) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def where(
    condition: Union[int, float, tensor_class],
    y: Union[int, float, tensor_class],
    z: Union[int, float, tensor_class],
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def ones_like(
    a: Union[int, float, tensor_class], allow_grad: py_bool = False
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def ones(shape: Union[int, Sequence[int]], allow_grad: py_bool = False) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def zeros_like(
    a: Union[int, float, tensor_class], allow_grad: py_bool = False
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def zeros(
    shape: Union[int, Sequence[int]], allow_grad: py_bool = False
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def full_like(
    a: tensor_class,
    x: Union[int, float, tensor_class],
    allow_grad: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def full(shape: Union[int, Sequence[int]], allow_grad: py_bool = False) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def concatenate(
    arrays: Sequence[Union[int, float, tensor_class]],
    axis: Optional[int] = 0,
    allow_grad: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def index_add(
    a: Union[int, float, tensor_class],
    indices: Union[int, float, tensor_class],
    b: Optional[Union[int, float, tensor_class]] = None,
):
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def isin(
    element: Union[int, float, tensor_class],
    test_elements: List[Union[int, float, tensor_class]],
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def unravel_index(
    indices: Union[int, float, tensor_class],
    shape: Sequence[int],
    allow_grad: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def vmap(
    fun: Callable[[tensor_class], tensor_class],
) -> Callable[[tensor_class], tensor_class]:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def take_along_axis(
    arr: tensor_class,
    indices: tensor_class,
    axis: Optional[int] = None,
    allow_grad: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def put_along_axis(
    arr: tensor_class,
    indices: tensor_class,
    values: Union[int, float, tensor_class],
    axis: Optional[int],
):
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def repeat(
    a: Union[int, float, tensor_class],
    repeats: Union[int, Sequence[int]],
    allow_grad: py_bool = False,
    axis: Optional[int] = None,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tile(
    A: Union[int, float, tensor_class],
    reps: Union[int, float, tensor_class],
    allow_grad: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def arange(*args: Union[int, float], allow_grad: py_bool = False) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def stack(
    arrays: Sequence[tensor_class], axis: Optional[int] = 0, allow_grad: py_bool = False
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def save(file, arr: Union[int, float, tensor_class]):
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def load(file, allow_grad: py_bool = False) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def choice(
    a: Union[int, Union[int, float, tensor_class]],
    size: Optional[Union[int, Sequence[int]]] = None,
    replace: py_bool = True,
    p: Optional[Union[int, float, tensor_class]] = None,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def rand(*dims: Optional[int], allow_grad: py_bool = False) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def randint(
    low: Union[int, Union[int, float, tensor_class]],
    high: Optional[Union[int, Union[int, float, tensor_class]]] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    allow_grad: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def randn(*dims: Optional[int], allow_grad: py_bool = False) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def binomial(
    n: Union[int, tensor_class[int]],
    p: Union[float, tensor_class[float]],
    size: Optional[Tuple[int]] = None,
    allow_grad: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def permutation(
    x: Union[int, tensor_class], allow_grad: py_bool = False
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def shuffle(x: tensor_class):
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def split(
    ary: tensor_class,
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0,
    allow_grad: py_bool = False,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tensor_shape(data: tensor_class) -> Tuple[int, ...]:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tensor_size(data: tensor_class) -> int:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tensor_ndim(data: tensor_class) -> int:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tensor_dtype(data: tensor_class) -> dtype:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def tensor_item(data: tensor_class) -> Any:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def repr(data: tensor_class) -> str:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def len(data: tensor_class) -> int:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def array_interface(data: tensor_class) -> dict[str, Any]:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


def array(
    data: tensor_class,
    dtype: Optional[dtype] = None,
    copy: Optional[py_bool] = None,
) -> tensor_class:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


class dtype:
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class float64(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class float32(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class float16(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class uint64(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class uint32(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class uint16(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class uint8(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class int64(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class int32(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class int16(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class int8(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


class bool(dtype):
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )


nan: Any


def as_numpy(a: tensor_class) -> np.array:
    raise NotImplementedError("Attempting to call default unimplemented Backend method")


class Backend:
    def __init__(self):
        raise NotImplementedError(
            f"Attempting to instantiate default unimplemented Backend class {self.__class__}"
        )
