from __future__ import annotations

from builtins import bool as py_bool
from typing import TYPE_CHECKING

from numpy import ndarray

import minidiff as md
from minidiff.backend import current_backend
from minidiff.utils import try_unwrap

if TYPE_CHECKING:
    from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

    import minidiff.typing as mdt
    from minidiff.topology import OpNode


# compute from left to right, dy/dw2 then dw2/dw1 to get dy/dw1 and finally dw1/dx to get dy/dx
# dy/dw2 would just be the loss gradient

# all tensors by default should not allow grad
# all tensors keep their allow_grad state whether in no_grad() or not; no_grad() just prevents any graph creation


class Tensor:
    def __init__(
        self,
        data: Optional[Union[int, float, mdt.BackendTensor]],
        allow_grad: py_bool = False,
        dtype: Optional[mdt.dtype] = None,
    ):
        data = try_unwrap(data)
        if data is None:
            data = current_backend.tensor_constructor([])
        if not isinstance(data, current_backend.tensor_class):
            data = current_backend.tensor_constructor(data)
        if dtype is not None:
            data = data.astype(dtype)
        self._data = data

        self._allow_grad = allow_grad
        self._iterator = None

        self.graph_refs = 0
        self.grad: Optional[Tensor] = None
        self.op_node: Optional[OpNode] = None

    # graphed means we are used in a gradient-tracked computation.
    # this means either there is some portion of the graph referencing us
    # or we are referencing some portion of the graph
    @property
    def graphed(self) -> py_bool:
        return self.graph_refs > 0 or self.op_node is not None

    # tensors not created by ops are leafs. this property is immutable
    @property
    def is_leaf(self) -> py_bool:
        return self.op_node is None

    @property
    def allow_grad(self) -> py_bool:
        return self._allow_grad

    @allow_grad.setter
    def allow_grad(self, allow_grad: py_bool):
        # turning off gradient tracking for intermediate tensors means gradients will definitely not propagate correctly
        # that means zeroed out gradients for an unclear reason, so it's better to fail fast
        if not allow_grad and not self.is_leaf:
            raise ValueError(
                "Turning off gradient tracking for intermediate tensors will almost always break chain rule in backprop"
            )

        if self._allow_grad == allow_grad:
            return

        # reset the gradient either way the state changes:
        # if we're enabling grad tracking then this should essentially do nothing
        # if we're disabling grad tracking this wipes the previous gradient from memory
        self.grad = None

        self._allow_grad = allow_grad

    @property
    def T(self) -> Tensor:
        return md.transpose(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        return current_backend.tensor_shape(self._data)

    @property
    def size(self) -> int:
        return current_backend.tensor_size(self._data)

    @property
    def ndim(self) -> int:
        return current_backend.tensor_ndim(self._data)

    @property
    def dtype(self) -> mdt.dtype:
        return current_backend.tensor_dtype(self._data)

    def as_numpy(self) -> ndarray:
        return current_backend.as_numpy(self._data)

    def backward(
        self,
        retain_grads: py_bool = False,
        cleanup_mode: Literal["keep", "prune", "destroy"] = "prune",
        allow_higher_order: py_bool = False,
        reset_grads: py_bool = True,
    ):
        # can't call backward if we're not tracking gradients or we have no gradient history
        if not self._allow_grad:
            return

        if self.is_leaf:
            return

        self.grad = md.ones_like(self, allow_grad=allow_higher_order)

        self.op_node.backward(
            self.grad,
            retain_grads=retain_grads,
            cleanup_mode=cleanup_mode,
            allow_higher_order=allow_higher_order,
            reset_grads=reset_grads,
        )

    # remove our subgraph from the whole graph
    def wipe(self):
        self.op_node = None

    # returns a view that does not have gradient history
    def detach(self, allow_grad: py_bool = False) -> Tensor:
        return Tensor(self._data, allow_grad=allow_grad)

    def ravel(self, order="C"):
        return md.ravel(self, order=order)

    def flatten(self, order="C"):
        return md.flatten(self, order=order)

    def astype(self, dtype: mdt.dtype):
        return md.astype(self, dtype)

    def transpose(self, axes: Optional[Union[int, Sequence[int]]] = None):
        return md.transpose(self, axes=axes)

    def item(self) -> Any:
        if self.size != 1:
            raise ValueError(
                "Only Tensors with a single element can be reduced to a Python scalar"
            )

        return current_backend.tensor_item(self._data)

    def sum(
        self,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: py_bool = False,
    ) -> Tensor:
        return md.sum(self, axis=axis, keepdims=keepdims)

    def copy(self) -> Tensor:
        return md.copy(self)

    def clip(
        self,
        a_min: Optional[Union[float, int]] = None,
        a_max: Optional[Union[float, int]] = None,
    ) -> Tensor:
        return md.clip(self, a_min=a_min, a_max=a_max)

    def reshape(self, shape: Union[int, Sequence[int]]) -> Tensor:
        return md.reshape(self, shape)

    def dot(self, other: mdt.TensorLike) -> Tensor:
        return md.dot(self, other)

    def matmul(self, other: mdt.TensorLike) -> Tensor:
        return md.matmul(self, other)

    def add(self, other: mdt.TensorLike) -> Tensor:
        return md.add(self, other)

    def multiply(self, other: mdt.TensorLike) -> Tensor:
        return md.multiply(self, other)

    def _graph_tracking(self):
        return self._allow_grad and md.grad_allowed_() and self.graphed

    def _validate_mutation(self):
        if self._graph_tracking():
            raise ValueError(
                "In-place operations can break computation graphs during backprop"
            )

    def __mod__(self, other: mdt.TensorLike) -> Tensor:
        return md.mod(self, other)

    def __imod__(self, other: mdt.TensorLike) -> Tensor:
        self._validate_mutation()

        self._data %= try_unwrap(other)

        return self

    def __matmul__(self, other: Tensor) -> Tensor:
        return md.matmul(self, other)

    def __imatmul__(self, other: Tensor) -> Tensor:
        self._validate_mutation()

        self._data @= other._data

        return self

    def __add__(self, other: mdt.TensorLike) -> Tensor:
        return md.add(self, other)

    def __radd__(self, other: mdt.TensorLike) -> Tensor:
        return md.add(other, self)

    def __iadd__(self, other: mdt.TensorLike) -> Tensor:
        self._validate_mutation()

        self._data += try_unwrap(other)

        return self

    def __sub__(self, other: mdt.TensorLike) -> Tensor:
        return md.subtract(self, other)

    def __rsub__(self, other: mdt.TensorLike) -> Tensor:
        return md.subtract(other, self)

    def __isub__(self, other: mdt.TensorLike) -> Tensor:
        self._validate_mutation()

        self._data -= try_unwrap(other)

        return self

    def __mul__(self, other: mdt.TensorLike) -> Tensor:
        return md.multiply(self, other)

    def __rmul__(self, other: mdt.TensorLike) -> Tensor:
        return md.multiply(other, self)

    def __imul__(self, other: mdt.TensorLike) -> Tensor:
        self._validate_mutation()

        self._data *= try_unwrap(other)

        return self

    def __truediv__(self, other: mdt.TensorLike) -> Tensor:
        return md.true_divide(self, other)

    def __rtruediv__(self, other: mdt.TensorLike) -> Tensor:
        return md.true_divide(other, self)

    def __itruediv__(self, other: mdt.TensorLike) -> Tensor:
        self._validate_mutation()

        self._data /= try_unwrap(other)

        return self

    def __floordiv__(self, other: mdt.TensorLike) -> Tensor:
        return md.floor_divide(self, other)

    def __rfloordiv__(self, other: mdt.TensorLike) -> Tensor:
        return md.floor_divide(other, self)

    def __ifloordiv__(self, other: mdt.TensorLike) -> Tensor:
        self._validate_mutation()

        self._data //= try_unwrap(other)

        return self

    def __pow__(self, other: mdt.TensorLike) -> Tensor:
        return md.power(self, other)

    def __rpow__(self, other: mdt.TensorLike) -> Tensor:
        return md.power(other, self)

    def __ipow__(self, other: mdt.TensorLike) -> Tensor:
        self._validate_mutation()

        self._data **= try_unwrap(other)

        return self

    def __neg__(self) -> Tensor:
        return -1 * self

    def __repr__(self) -> str:
        return current_backend.repr(self._data)

    def __len__(self) -> int:
        return current_backend.len(self._data)

    def __getitem__(self, key: Any) -> Tensor:
        return md.getitem(self, key)

    def __setitem__(self, key: Any, val: mdt.TensorLike):
        self._validate_mutation()

        self._data[try_unwrap(key)] = try_unwrap(val)

    def __gt__(self, value: mdt.TensorLike) -> Tensor:
        return md.greater(self, value)

    def __ge__(self, value: mdt.TensorLike) -> Tensor:
        return md.greater_equal(self, value)

    def __lt__(self, value: mdt.TensorLike) -> Tensor:
        return md.less(self, value)

    def __le__(self, value: mdt.TensorLike) -> Tensor:
        return md.less_equal(self, value)

    def __eq__(self, value: mdt.TensorLike) -> Tensor:
        return md.equal(self, value)

    def __ne__(self, value: mdt.TensorLike) -> Tensor:
        return md.not_equal(self, value)

    def __and__(self, value: mdt.TensorLike) -> Tensor:
        return md.logical_and(self, value)

    def __or__(self, value: mdt.TensorLike) -> Tensor:
        return md.logical_or(self, value)

    def __not__(self, value: mdt.TensorLike) -> Tensor:
        return md.logical_not(self, value)

    def __xor__(self, value: mdt.TensorLike) -> Tensor:
        return md.logical_xor(self, value)

    def __invert__(self) -> Tensor:
        return md.invert(self)

    def __iter__(self) -> TensorIterator:
        if self._iterator is None:
            data_size = current_backend.tensor_size(self._data)
            self._iterator = TensorIterator(
                self,
                len(self) if data_size > 1 else data_size,
            )
        return self._iterator

    # numpy array specification requirements:
    @property
    def __array_interface__(self) -> Dict[str, Any]:
        return current_backend.array_interface(self._data)

    def __array__(
        self,
        dtype: Optional[mdt.dtype] = None,
        copy: Optional[py_bool] = None,
    ) -> ndarray:
        return current_backend.array(self._data, dtype=dtype, copy=copy)


class TensorIterator:
    def __init__(self, data: Tensor, length: int):
        self.data = data
        self.length = length
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tensor:
        if self.index >= self.length:
            raise StopIteration
        item = self.data[self.index]
        self.index += 1
        return item
