from __future__ import annotations

import contextvars
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import cupy as np  # type: ignore
    import cupy.typing as npt  # type: ignore
except ImportError:
    import numpy as np
    import numpy.typing as npt

import minidiff as md
import minidiff.typing as mdt

_allow_grad = contextvars.ContextVar("allow_grad", default=True)


class no_grad:
    def __enter__(self):
        self.prev = _allow_grad.get()
        set_allow_grad(False)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev)


def set_allow_grad(allow):
    _allow_grad.set(allow)


def grad_allowed_():
    return _allow_grad.get()


# compute from left to right, dy/dw2 then dw2/dw1 to get dy/dw1 and finally dw1/dx to get dy/dx
# dy/dw2 would just be the loss gradient

# all tensors by default should not allow grad
# all tensors keep their allow_grad state whether in no_grad() or not; no_grad() just prevents any graph creation


class Tensor:
    def __init__(
        self,
        data: "np.ArrayLike",
        allow_grad: bool = False,
        dtype: Optional[np.dtype] = None,
        func_node: Optional["md.topology.FuncNode"] = None,
    ):
        if isinstance(data, np.ndarray):
            self._data = data
        else:
            self._data = np.array(data)

        if dtype is not None:
            self._data = self._data.astype(dtype)

        self._allow_grad = allow_grad
        # tensors not created by ops are leafs. this property is immutable
        self._func_node = func_node
        # graphed means we are used in a gradient-tracked computation.
        self.graphed = False
        # don't store gradients unless we are user-created.
        self.grad = (
            zeros_like(self, allow_grad=False) if self.is_leaf and allow_grad else None
        )

    @property
    def func_node(self) -> "md.topology.FuncNode":
        return self._func_node

    @func_node.setter
    def func_node(self, func_node: "md.topology.FuncNode"):
        # if we're on a graph, and we're a leaf that's trying to assign itself a node,
        # then fail because leafs, by definition, have no gradient history and therefore
        # cannot possess nodes
        if self.graphed and (self.is_leaf and func_node is not None):
            raise ValueError("leaf tensors cannot possess func_nodes")

        self._func_node = func_node

    # we're a leaf if we have no gradient history
    # whether we're part of a gradient-tracked computation or not.
    @property
    def is_leaf(self) -> bool:
        return self.func_node is None

    @property
    def allow_grad(self) -> bool:
        return self._allow_grad

    @allow_grad.setter
    def allow_grad(self, allow_grad: bool):
        # if we're trying to turn off gradient tracking while we're graphed and an intermediate tensor, then error
        if not allow_grad and (self.graphed and not self.is_leaf):
            raise ValueError(
                "Tensors can only stop tracking gradients if they are not part of a computational graph or are a leaf"
            )

        if self._allow_grad == allow_grad:
            return

        # any tensors who don't allow gradient-tracking don't track their gradients.
        # intermediate non-leaf tensors do not have gradients because we don't care
        if not allow_grad or not self.is_leaf:
            self.grad = None
        else:
            self.grad = zeros_like(self, allow_grad=False)

        self._allow_grad = allow_grad

    @property
    def T(self) -> "Tensor":
        return md.transpose(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def toposort(self) -> List["Tensor"]:
        seen = set()
        traversal_path = []

        # topologically sort:
        # step through the graph starting from the output tensor (self)
        # go all the way down to the leaf tensors, skipping tensors we've already seen
        # after getting all the way to the base, finally push ourselves onto the stack
        # rinse and repeat for the input tensors, their input tensors, etc.
        def dfs(op_output):
            if id(op_output) in seen:
                return
            seen.add(id(op_output))
            if not op_output.is_leaf:
                root = op_output.func_node
                for op_input in root.input_tensors:
                    dfs(op_input)
            traversal_path.append(op_output)

        dfs(self)

        return traversal_path

    # this does the actual advertised reverse-mode automatic differentiation.
    # I mostly just referenced this Wikipedia page: https://en.wikipedia.org/wiki/Automatic_differentiation
    def backward(self, retain_graph: bool = False, retain_grads: bool = False):
        # can't call backward if we're not tracking gradients or we have no gradient history
        if not self.allow_grad:
            return

        if self.is_leaf:
            return

        traversal_path = self.toposort()

        self.grad = ones_like(self, allow_grad=False)

        for tensor in reversed(traversal_path):
            # leaf tensors don't have any input tensors to update, so skip
            if tensor.is_leaf:
                continue
            n = tensor.func_node
            n_grad = tensor.grad
            n.update_grads(n_grad)
            # we're only temporarily storing grads, so we need to remove any references when
            # we're done for the sake of memory
            if not retain_grads:
                tensor.grad = None
            if not retain_graph:
                tensor.wipe()

    # destroy our portion of the graph
    def wipe(self):
        self.graphed = False
        self.func_node = None

    # returns a copy that does not track gradients
    def detach(self, allow_grad: bool = False) -> "Tensor":
        detached = Tensor(self._data.copy(), allow_grad=allow_grad)
        return detached

    def astype(self, dtype, **kwargs):
        return md.astype(self, dtype, **kwargs)

    def transpose(self, axes=None):
        return md.transpose(self, axes=axes)

    def item(self) -> np.ScalarType:
        if self.size != 1:
            raise ValueError(
                "only Tensors with a single element can be reduced to a Python scalar"
            )

        return self._data.item()

    def sum(self, **kwargs) -> "Tensor":
        return md.sum(self, **kwargs)

    def copy(self, **kwargs) -> "Tensor":
        return md.copy(self, **kwargs)

    def clip(
        self,
        a_min: Optional[Union[float, int]] = None,
        a_max: Optional[Union[float, int]] = None,
    ) -> "Tensor":
        return md.clip(self, a_min=a_min, a_max=a_max)

    def reshape(self, shape: Sequence[int], **kwargs) -> "Tensor":
        return md.reshape(self, shape, **kwargs)

    def dot(self, other: "Tensor", **kwargs) -> "Tensor":
        return md.matmul(self, other, **kwargs)

    def add(self, other: mdt.TensorLike, **kwargs) -> "Tensor":
        return md.add(self, other, **kwargs)

    def multiply(self, other: mdt.TensorLike, **kwargs) -> "Tensor":
        return md.multiply(self, other, **kwargs)

    def _allow_mutation(self):
        return not (self.allow_grad and md.grad_allowed_() and self.graphed)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return md.matmul(self, other)

    def __imatmul__(self, other: "Tensor") -> "Tensor":
        if not self._allow_mutation():
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        self._data = self._data @ other._data
        return self

    def __add__(self, other: mdt.TensorLike) -> "Tensor":
        return md.add(self, other)

    def __radd__(self, other: mdt.TensorLike) -> "Tensor":
        return md.add(other, self)

    def __iadd__(self, other: mdt.TensorLike) -> "Tensor":
        if not self._allow_mutation():
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.add(self._data, other._data, casting="safe")
        else:
            self._data = np.add(self._data, other, casting="safe")

        return self

    def __sub__(self, other: mdt.TensorLike) -> "Tensor":
        return md.subtract(self, other)

    def __rsub__(self, other: mdt.TensorLike) -> "Tensor":
        return md.subtract(other, self)

    def __isub__(self, other: mdt.TensorLike) -> "Tensor":
        if not self._allow_mutation():
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.subtract(self._data, other._data, casting="safe")
        else:
            self._data = np.subtract(self._data, other, casting="safe")

        return self

    def __mul__(self, other: mdt.TensorLike) -> "Tensor":
        return md.multiply(self, other)

    def __rmul__(self, other: mdt.TensorLike) -> "Tensor":
        return md.multiply(other, self)

    def __imul__(self, other: mdt.TensorLike) -> "Tensor":
        if not self._allow_mutation():
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.multiply(self._data, other._data, casting="safe")
        else:
            self._data = np.multiply(self._data, other, casting="safe")

        return self

    def __truediv__(self, other: mdt.TensorLike) -> "Tensor":
        return md.true_divide(self, other)

    def __rtruediv__(self, other: mdt.TensorLike) -> "Tensor":
        return md.true_divide(other, self)

    def __itruediv__(self, other: mdt.TensorLike) -> "Tensor":
        if not self._allow_mutation():
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.true_divide(self._data, other._data, casting="safe")
        else:
            self._data = np.true_divide(self._data, other, casting="safe")

        return self

    def __floordiv__(self, other: mdt.TensorLike) -> "Tensor":
        return md.floor_divide(self, other)

    def __rfloordiv__(self, other: mdt.TensorLike) -> "Tensor":
        return md.floor_divide(other, self)

    def __ifloordiv__(self, other: mdt.TensorLike) -> "Tensor":
        if not self._allow_mutation():
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.floor_divide(self._data, other._data, casting="safe")
        else:
            self._data = np.floor_divide(self._data, other, casting="safe")

        return self

    def __pow__(self, other: mdt.TensorLike) -> "Tensor":
        return md.power(self, other)

    def __rpow__(self, other: mdt.TensorLike) -> "Tensor":
        return md.power(other, self)

    def __ipow__(self, other: mdt.TensorLike) -> "Tensor":
        if not self._allow_mutation():
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.power(self._data, other._data, casting="safe")
        else:
            self._data = np.power(self._data, other, casting="safe")

        return self

    def __neg__(self) -> "Tensor":
        return -1 * self

    def __repr__(self) -> str:
        return self._data.__repr__()

    def __len__(self) -> int:
        return self._data.__len__()

    def __getitem__(self, key: Any) -> "Tensor":
        return md.getitem(self, key)

    def __setitem__(self, key: Any, val: mdt.TensorLike):
        if not self._allow_mutation():
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(val, Tensor):
            self._data[key] = val._data
        else:
            self._data[key] = val

    def __gt__(self, value: mdt.TensorLike) -> "Tensor":
        return md.greater(self, value)

    def __ge__(self, value: mdt.TensorLike) -> "Tensor":
        return md.greater_equal(self, value)

    def __lt__(self, value: mdt.TensorLike) -> "Tensor":
        return md.less(self, value)

    def __le__(self, value: mdt.TensorLike) -> "Tensor":
        return md.less_equal(self, value)

    def __eq__(self, value: mdt.TensorLike) -> "Tensor":
        return md.equal(self, value)

    def __ne__(self, value: mdt.TensorLike) -> "Tensor":
        return md.not_equal(self, value)

    def __and__(self, value: mdt.TensorLike) -> "Tensor":
        return md.logical_and(self, value)

    def __or__(self, value: mdt.TensorLike) -> "Tensor":
        return md.logical_or(self, value)

    def __not__(self, value: mdt.TensorLike) -> "Tensor":
        return md.logical_not(self, value)

    def __xor__(self, value: mdt.TensorLike) -> "Tensor":
        return md.logical_xor(self, value)

    # numpy array specification requirements:
    @property
    def __array_interface__(self) -> Dict[str, Any]:
        return self._data.__array_interface__

    def __array__(
        self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None
    ) -> npt.NDArray[Any]:
        if dtype is not None and dtype != self.dtype:
            if not copy:
                raise ValueError("attempted cast, but copies are not permitted")
            return self._data.astype(dtype=dtype)
        if copy:
            return self._data.copy()
        return self._data


def ones_like(a: mdt.TensorLike, allow_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.ones_like(a, **kwargs), allow_grad=allow_grad)


def ones(shape: Sequence[int], allow_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.ones(shape, **kwargs), allow_grad=allow_grad)


def zeros_like(a: mdt.TensorLike, allow_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.zeros_like(a, **kwargs), allow_grad=allow_grad)


def zeros(shape: Sequence[int], allow_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.zeros(shape, **kwargs), allow_grad=allow_grad)


def full_like(
    a: Tensor, x: mdt.TensorLike, allow_grad: bool = False, **kwargs
) -> Tensor:
    return Tensor(np.full_like(a, x, **kwargs), allow_grad=allow_grad)


def full(shape: Sequence[int], allow_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.full(shape, **kwargs), allow_grad=allow_grad)


def unravel_index(
    indices: mdt.TensorLike, shape: Sequence[int], allow_grad: bool = False, **kwargs
) -> Tensor:
    return Tensor(np.unravel_index(indices, shape, **kwargs), allow_grad=allow_grad)


def repeat(
    a: mdt.TensorLike,
    repeats: Union[int, Sequence[int]],
    allow_grad: bool = False,
    axis: Optional[int] = None,
) -> Tensor:
    return Tensor(np.repeat(a, repeats, axis=axis), allow_grad=allow_grad)


def tile(A: mdt.TensorLike, reps: mdt.TensorLike, allow_grad: bool = False) -> Tensor:
    return Tensor(np.tile(A, reps), allow_grad=allow_grad)


def arange(*args: Union[int, float], allow_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.arange(*args, **kwargs), allow_grad=allow_grad)
