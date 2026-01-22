from __future__ import annotations

from contextvars import ContextVar
from builtins import bool as py_bool
from typing import TYPE_CHECKING

from numpy import ndarray

import minidiff as md
from minidiff.backend import current_backend
import minidiff.caching as mdc

if TYPE_CHECKING:
    from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

    import minidiff.typing as mdt
    from minidiff.topology import OpNode


_allow_grad = ContextVar("allow_grad", default=True)
_allow_new_grads = ContextVar("allow_new_grads", default=True)


class disable_new_grads:
    def __enter__(self):
        self.prev_allow_grad = _allow_grad.get()
        self.prev_allow_new_grads = _allow_new_grads.get()
        set_allow_grad(False)
        set_allow_new_grads(False)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev_allow_grad)
        set_allow_new_grads(self.prev_allow_new_grads)


class no_grad:
    def __enter__(self):
        self.prev = _allow_grad.get()
        set_allow_grad(False)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev)


class enable_grad:
    def __init__(self, enable: py_bool):
        self.enable = enable

    def __enter__(self):
        self.prev = _allow_grad.get()
        set_allow_grad(self.enable)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev)


def set_allow_new_grads(allow: py_bool):
    _allow_new_grads.set(allow)


def new_grads_allowed_() -> py_bool:
    return _allow_new_grads.get()


def set_allow_grad(allow: py_bool):
    _allow_grad.set(allow)


def grad_allowed_() -> py_bool:
    return _allow_grad.get()


def try_unwrap(t: Any):
    if isinstance(t, Tensor):
        return t._data
    elif isinstance(t, tuple):
        return tuple(try_unwrap(x) for x in t)
    elif isinstance(t, list):
        return [try_unwrap(x) for x in t]
    elif isinstance(t, dict):
        return {key: try_unwrap(val) for key, val in t.items()}
    else:
        return t


# compute from left to right, dy/dw2 then dw2/dw1 to get dy/dw1 and finally dw1/dx to get dy/dx
# dy/dw2 would just be the loss gradient

# all tensors by default should not allow grad
# all tensors keep their allow_grad state whether in no_grad() or not; no_grad() just prevents any graph creation


class Tensor:
    def __init__(
        self,
        data: Optional[Union[int, float, current_backend.tensor_class]],
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

    def toposort(self) -> List[Tensor]:
        seen = set()
        traversal_path = []
        stack = [self]
        parents = []

        while len(stack) != 0:
            tensor = stack.pop()
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))

            node = tensor.op_node
            all_children_visited = True

            # append all children to the stack so they will be processed next
            # (DFS)
            if node is not None:
                for t in node.tensor_inputs:
                    if id(t) in seen:
                        continue
                    all_children_visited = False
                    stack.append(t)

            # if not all children have been seen, at least one is appended to the stack
            # therefore the current tensor must be the next parent
            # this is the post-order bit
            if not all_children_visited:
                parents.append(tensor)
                continue

            # all children were visited so that means we need to hop up a level on the graph (go to the parent)
            # if there are no parents left, that means the current tensor is the output tensor
            traversal_path.append(tensor)
            if len(parents) != 0:
                parent = parents.pop()
                seen.remove(id(parent))
                stack.append(parent)

        return traversal_path

    # this does the actual advertised reverse-mode automatic differentiation.
    # I mostly just referenced this Wikipedia page: https://en.wikipedia.org/wiki/Automatic_differentiation
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

        if cleanup_mode not in ["keep", "prune", "destroy"]:
            cleanup_mode = "prune"

        # computing higher order derivatives means partially re-traversing the subgraph for whichever variable
        # we're computing the higher order derivative of, so the graph needs to remain.
        # in accumulating gradients when calling backward() the second time, gradients from intermediates
        # will almost always be necessary so those have to be kept in memory too
        if allow_higher_order:
            retain_grads = True
            if cleanup_mode == "destroy":
                cleanup_mode = "prune"

        if mdc.currently_caching():
            traversal_indices = mdc.indices_for_tensor(self)
            full_tree = self.op_node.tree + [self]
            traversal_path = [full_tree[index] for index in traversal_indices]
        else:
            traversal_path = self.toposort()

        if reset_grads:
            for tensor in traversal_path:
                tensor.grad = None

        self.grad = ones_like(self)

        with enable_grad(allow_higher_order):
            for tensor in reversed(traversal_path):
                # leaf tensors don't have any input tensors to update, so skip
                if tensor.is_leaf:
                    continue
                # this should never be None since the final gradient (self's gradient) is manually set to ones
                # first iteration updates input tensors who now have non-None grads too
                # this continues for their input tensors, and those tensor's inputs, and so on and so forth
                grad = tensor.grad
                grad.allow_grad = allow_higher_order
                node = tensor.op_node
                node.update_grads(grad)
                # we're only temporarily storing grads
                # so we need to remove any references when we're done to save memory
                if not retain_grads:
                    tensor.grad = None

                if cleanup_mode == "keep":
                    continue

                if cleanup_mode == "destroy":
                    tensor.wipe()
                    continue

                if tensor.graph_refs > 0:
                    continue

                for child in node.tensor_inputs:
                    child.graph_refs -= 1

                tensor.wipe()

    # destroy our portion of the graph
    def wipe(self):
        self.op_node = None

    # returns a view that does not have gradient history
    def detach(self, allow_grad: py_bool = False) -> Tensor:
        return Tensor(self._data, allow_grad=allow_grad)

    def ravel(self, order="C"):
        return md.ravel(self, order=order)

    def flatten(self, order="C"):
        return md.flatten(self, order=order)

    def astype(self, dtype: mdt.dtype, **kwargs):
        return md.astype(self, dtype, **kwargs)

    def transpose(self, axes: Optional[Union[int, Sequence[int]]] = None):
        return md.transpose(self, axes=axes)

    def item(self) -> Any:
        if self.size != 1:
            raise ValueError(
                "Only Tensors with a single element can be reduced to a Python scalar"
            )

        return current_backend.tensor_item(self._data)

    def sum(self, **kwargs) -> Tensor:
        return md.sum(self, **kwargs)

    def copy(self, **kwargs) -> Tensor:
        return md.copy(self, **kwargs)

    def clip(
        self,
        a_min: Optional[Union[float, int]] = None,
        a_max: Optional[Union[float, int]] = None,
    ) -> Tensor:
        return md.clip(self, a_min=a_min, a_max=a_max)

    def reshape(self, shape: Union[int, Sequence[int]], **kwargs) -> Tensor:
        return md.reshape(self, shape, **kwargs)

    def dot(self, other: mdt.TensorLike, **kwargs) -> Tensor:
        return md.dot(self, other, **kwargs)

    def matmul(self, other: mdt.TensorLike, **kwargs) -> Tensor:
        return md.matmul(self, other, **kwargs)

    def add(self, other: mdt.TensorLike, **kwargs) -> Tensor:
        return md.add(self, other, **kwargs)

    def multiply(self, other: mdt.TensorLike, **kwargs) -> Tensor:
        return md.multiply(self, other, **kwargs)

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

        self._data = self._data % try_unwrap(other)

        return self

    def __matmul__(self, other: Tensor) -> Tensor:
        return md.matmul(self, other)

    def __imatmul__(self, other: Tensor) -> Tensor:
        self._validate_mutation()

        self._data = self._data @ other._data

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
                self._data,
                len(self) if data_size > 1 else data_size,
            )
        return self._iterator

    # numpy array specification requirements:
    @property
    def __array_interface__(self) -> Dict[str, Any]:
        return current_backend.array_interface(self._data)

    def __array__(
        self,
        dtype: Optional[current_backend.dtype] = None,
        copy: Optional[py_bool] = None,
    ) -> ndarray:
        return current_backend.array(self._data, dtype=dtype, copy=copy)


class TensorIterator:
    def __init__(self, data, length):
        self.data = data
        self.length = length
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        item = self.data[self.index]
        self.index += 1
        return item


def ones_like(a: mdt.TensorLike, allow_grad: py_bool = False, **kwargs) -> Tensor:
    a = try_unwrap(a)

    return Tensor(current_backend.ones_like(a, **kwargs), allow_grad=allow_grad)


def ones(
    shape: Union[int, Sequence[int]], allow_grad: py_bool = False, **kwargs
) -> Tensor:
    return Tensor(current_backend.ones(shape, **kwargs), allow_grad=allow_grad)


def zeros_like(a: mdt.TensorLike, allow_grad: py_bool = False, **kwargs) -> Tensor:
    a = try_unwrap(a)

    return Tensor(current_backend.zeros_like(a, **kwargs), allow_grad=allow_grad)


def zeros(
    shape: Union[int, Sequence[int]], allow_grad: py_bool = False, **kwargs
) -> Tensor:
    return Tensor(current_backend.zeros(shape, **kwargs), allow_grad=allow_grad)


def full_like(
    a: Tensor, x: mdt.TensorLike, allow_grad: py_bool = False, **kwargs
) -> Tensor:
    a = try_unwrap(a)
    x = try_unwrap(x)

    return Tensor(current_backend.full_like(a, x, **kwargs), allow_grad=allow_grad)


def full(
    shape: Union[int, Sequence[int]], allow_grad: py_bool = False, **kwargs
) -> Tensor:
    return Tensor(current_backend.full(shape, **kwargs), allow_grad=allow_grad)


def concatenate(
    arrays: Sequence[mdt.TensorLike],
    axis: Optional[int] = 0,
    allow_grad: py_bool = False,
) -> Tensor:
    arrays = try_unwrap(arrays)
    return Tensor(current_backend.concatenate(arrays, axis=axis), allow_grad=allow_grad)


def index_add(
    a: mdt.TensorLike, indices: mdt.TensorLike, b: Optional[mdt.TensorLike] = None
):
    a = try_unwrap(a)
    indices = try_unwrap(indices)
    b = try_unwrap(b)

    current_backend.index_add(a, indices, b)


def isin(
    element: mdt.TensorLike, test_elements: List[mdt.TensorLike], **kwargs
) -> py_bool:
    element = try_unwrap(element)
    test_elements = try_unwrap(test_elements)

    return current_backend.isin(element, test_elements, **kwargs)


def unravel_index(
    indices: mdt.TensorLike, shape: Sequence[int], allow_grad: py_bool = False, **kwargs
) -> Tensor:
    indices = try_unwrap(indices)

    return Tensor(
        current_backend.unravel_index(indices, shape, **kwargs), allow_grad=allow_grad
    )


def vmap(fun: mdt.UnaryFunc) -> mdt.UnaryFunc:
    def backend_func(arr, *args, **kwargs):
        args = [Tensor(x) for x in args]
        kwargs = {key: Tensor(val) for key, val in kwargs.items()}
        val = fun(
            Tensor(arr),
            *args,
            **kwargs,
        )
        return val._data

    vmap_func = current_backend.vmap(backend_func)

    def wrapper(*args, **kwargs) -> Tensor:
        args = try_unwrap(args)
        kwargs = try_unwrap(kwargs)
        return Tensor(vmap_func(*args, **kwargs))

    return wrapper


def take_along_axis(
    arr: Tensor,
    indices: Tensor,
    axis: Optional[int] = None,
    allow_grad: py_bool = False,
) -> Tensor:
    arr = arr._data
    indices = indices._data

    return Tensor(
        current_backend.take_along_axis(arr, indices, axis=axis), allow_grad=allow_grad
    )


def put_along_axis(
    arr: Tensor,
    indices: Tensor,
    values: mdt.TensorLike,
    axis: Optional[int],
) -> Tensor:
    arr = arr._data
    indices = indices._data
    values = try_unwrap(values)

    current_backend.put_along_axis(arr, indices, values, axis)


def repeat(
    a: mdt.TensorLike,
    repeats: Union[int, Sequence[int]],
    allow_grad: py_bool = False,
    axis: Optional[int] = None,
) -> Tensor:
    a = try_unwrap(a)

    return Tensor(current_backend.repeat(a, repeats, axis=axis), allow_grad=allow_grad)


def tile(
    A: mdt.TensorLike, reps: mdt.TensorLike, allow_grad: py_bool = False
) -> Tensor:
    A = try_unwrap(A)
    reps = try_unwrap(reps)

    return Tensor(current_backend.tile(A, reps), allow_grad=allow_grad)


def arange(*args: Union[int, float], allow_grad: py_bool = False, **kwargs) -> Tensor:
    return Tensor(current_backend.arange(*args, **kwargs), allow_grad=allow_grad)


def stack(arrays: Sequence[Tensor], allow_grad: py_bool = False, **kwargs) -> Tensor:
    arrays = [x._data for x in arrays]

    return Tensor(current_backend.stack(arrays, **kwargs), allow_grad=allow_grad)


def save(file, arr: mdt.TensorLike, **kwargs):
    arr = arr._data

    current_backend.save(file, arr, **kwargs)


def load(file, allow_grad: py_bool = False, **kwargs) -> Tensor:
    return Tensor(current_backend.load(file, **kwargs), allow_grad=allow_grad)


def choice(
    a: Union[int, mdt.TensorLike],
    size: Optional[Union[int, Sequence[int]]] = None,
    replace: py_bool = True,
    p: Optional[mdt.TensorLike] = None,
) -> Tensor:
    a = try_unwrap(a)
    p = try_unwrap(p)

    return Tensor(current_backend.choice(a, size=size, replace=replace, p=p))


def rand(*dims: Optional[int], allow_grad: py_bool = False) -> Tensor:
    return Tensor(current_backend.rand(*dims), allow_grad=allow_grad)


def randint(
    low: Union[int, mdt.TensorLike],
    high: Optional[Union[int, mdt.TensorLike]] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    allow_grad: py_bool = False,
) -> Tensor:
    low = try_unwrap(low)
    high = try_unwrap(high)

    return Tensor(
        current_backend.randint(low, high=high, size=size), allow_grad=allow_grad
    )


def randn(*dims: Optional[int], allow_grad: py_bool = False) -> Tensor:
    return Tensor(current_backend.randn(*dims), allow_grad=allow_grad)


def binomial(
    n: Union[int, Tensor[int]],
    p: Union[float, Tensor[float]],
    size: Optional[Tuple[int]] = None,
    allow_grad: py_bool = False,
) -> Tensor:
    n = try_unwrap(n)
    p = try_unwrap(p)

    return Tensor(current_backend.binomial(n, p, size=size), allow_grad=allow_grad)


def permutation(x: Union[int, Tensor], allow_grad: py_bool = False) -> Tensor:
    x = try_unwrap(x)

    return Tensor(current_backend.permutation(x), allow_grad=allow_grad)


def shuffle(x: Tensor):
    current_backend.shuffle(x._data)


def split(
    ary: Tensor,
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0,
    allow_grad: py_bool = False,
) -> Tensor:
    ary = ary._data
    indices_or_sections = try_unwrap(indices_or_sections)

    backend_output = current_backend.split(ary, indices_or_sections, axis=axis)
    output = [None] * len(backend_output)

    for i, section in enumerate(backend_output):
        output[i] = Tensor(section, allow_grad=allow_grad)

    return output


dtypes = [
    float64 := current_backend.float64,
    float32 := current_backend.float32,
    float16 := current_backend.float16,
    uint64 := current_backend.uint64,
    uint32 := current_backend.uint32,
    uint16 := current_backend.uint16,
    uint8 := current_backend.uint8,
    int64 := current_backend.int64,
    int32 := current_backend.int32,
    int16 := current_backend.int16,
    int8 := current_backend.int8,
    bool := current_backend.bool,
]
newaxis = None
