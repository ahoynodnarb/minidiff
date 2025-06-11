try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import minidiff as md
import contextvars

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
# all tensors keep their allow_grad state whether in no_grad() or not, no_grad() just prevents any graph creation


class Tensor:
    def __init__(self, data, allow_grad=False, dtype=None, func_node=None):
        if isinstance(data, np.ndarray):
            self._data = data.astype(dtype)
        else:
            self._data = np.array(data, dtype=dtype)

        self._allow_grad = allow_grad
        # tensors not created by ops are leafs. this property is immutable
        self._func_node = func_node
        # don't store gradients unless we are user-created.
        self.graphed = False
        self.grad = (
            zeros_like(self, allow_grad=False) if self.is_leaf and allow_grad else None
        )

    @property
    def func_node(self):
        return self._func_node

    @func_node.setter
    def func_node(self, func_node):
        # if we're on a graph, and we're a leaf that's trying to assign itself a node,
        # then fail because leafs, by definition, have no gradient history and therefore
        # cannot possess nodes
        if self.graphed and (self.is_leaf and func_node is not None):
            raise ValueError("leaf tensors cannot possess func_nodes")

        self._func_node = func_node

    # we're a leaf if we have no gradient history and we are tracking gradients
    @property
    def is_leaf(self):
        return self.is_graph_source and self.allow_grad

    @property
    def is_graph_source(self):
        return self.func_node is None

    @property
    def allow_grad(self):
        return self._allow_grad

    @allow_grad.setter
    def allow_grad(self, allow_grad):
        if not allow_grad and self.graphed:
            raise ValueError(
                "Tensors can only stop tracking gradients if they are not part of a computational graph"
            )

        if self._allow_grad == allow_grad:
            return

        # any tensors who don't allow gradient-tracking don't track their gradients
        # intermediate non-leaf tensors do not have gradients because we don't care
        if not allow_grad or not self.is_leaf:
            self.grad = None
        else:
            self.grad = zeros_like(self, allow_grad=False)

        self._allow_grad = allow_grad

    @property
    def T(self):
        return self._data.T

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    def toposort(self):
        seen = set()
        traversal_path = []

        # topologically sort
        def dfs(tensor):
            root = tensor.func_node
            if root is None or id(root) in seen:
                return
            # temporarily maintain gradients only during backward pass
            if not tensor.is_leaf:
                tensor.grad = zeros_like(tensor, allow_grad=False)
            seen.add(id(root))
            for input_tensor in root.input_tensors:
                dfs(input_tensor)
            traversal_path.append(tensor)

        dfs(self)

        return traversal_path

    def backward(self, retain_graph=False, retain_grads=False):
        if not self.allow_grad:
            return

        if self.is_leaf:
            return

        traversal_path = self.toposort()

        self.grad = ones_like(self, allow_grad=False)

        for tensor in reversed(traversal_path):
            n = tensor.func_node
            n_grad = tensor.grad
            n.update_grads(n_grad)
            # we're only temporarily storing grads, so we need to remove any references when
            # we're done for the sake of memory
            if not tensor.is_leaf and not retain_grads:
                tensor.grad = None
            if not retain_graph:
                self.wipe()

    def wipe(self):
        self.graphed = False
        if (func_node := self.func_node) is None:
            return
        for input_tensor in func_node.input_tensors:
            input_tensor.graphed = False

    def item(self):
        if self.size != 1:
            raise ValueError(
                "only Tensors with a single element can be reduced to a Python scalar"
            )

        return self._data.item()

    # def flatten(self, **kwargs):
    #     return Tensor(self._data.flatten(**kwargs), allow_grad=self.allow_grad)

    def copy(self, **kwargs):
        return md.copy(self, **kwargs)

    def clip(self, a_min=None, a_max=None):
        return md.clip(self, a_min=a_min, a_max=a_max)

    def reshape(self, shape, **kwargs):
        return md.reshape(self, shape, **kwargs)

    def dot(self, other, **kwargs):
        return md.matmul(self, other, **kwargs)

    def add(self, other, **kwargs):
        return md.add(self, other, **kwargs)

    def multiply(self, other, **kwargs):
        return md.multiply(self, other, **kwargs)

    def __matmul__(self, other):
        return md.matmul(self, other)

    def __imatmul__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        self._data = self._data @ other._data
        return self

    def __add__(self, other):
        return md.add(self, other)

    def __radd__(self, other):
        return md.add(other, self)

    def __iadd__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.add(self._data, other._data, casting="safe")
        else:
            self._data = np.add(self._data, other, casting="safe")

        return self

    def __sub__(self, other):
        return md.subtract(self, other)

    def __rsub__(self, other):
        return md.subtract(other, self)

    def __isub__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.subtract(self._data, other._data, casting="safe")
        else:
            self._data = np.subtract(self._data, other, casting="safe")

        return self

    def __mul__(self, other):
        return md.multiply(self, other)

    def __rmul__(self, other):
        return md.multiply(other, self)

    def __imul__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.multiply(self._data, other._data, casting="safe")
        else:
            self._data = np.multiply(self._data, other, casting="safe")

        return self

    def __truediv__(self, other):
        return md.true_divide(self, other)

    def __rtruediv__(self, other):
        return md.true_divide(other, self)

    def __itruediv__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.true_divide(self._data, other._data, casting="safe")
        else:
            self._data = np.true_divide(self._data, other, casting="safe")

        return self

    def __floordiv__(self, other):
        return md.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return md.floor_divide(other, self)

    def __ifloordiv__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.floor_divide(self._data, other._data, casting="safe")
        else:
            self._data = np.floor_divide(self._data, other, casting="safe")

        return self

    def __pow__(self, other):
        return md.power(self, other)

    def __rpow__(self, other):
        return md.power(other, self)

    def __ipow__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = np.power(self._data, other._data, casting="safe")
        else:
            self._data = np.power(self._data, other, casting="safe")

        return self

    def __neg__(self):
        return -1 * self

    def __repr__(self):
        return self._data.__repr__()

    def __len__(self):
        return self._data.__len__()

    def __getitem__(self, key):
        # this should be an op actually
        return md.s_(self, key)

    def __setitem__(self, key, val):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )
        if self.graphed:
            raise ValueError(
                "mutating tensors is not allowed if the tensor is on a computational graph"
            )
        if isinstance(val, Tensor):
            self._data[key] = val._data
        else:
            self._data[key] = val

    def __gt__(self, value):
        return md.greater(self, value)

    def __ge__(self, value):
        return md.greater_equal(self, value)

    def __lt__(self, value):
        return md.less(self, value)

    def __le__(self, value):
        return md.less_equal(self, value)

    def __eq__(self, value):
        return md.equal(self, value)

    def __ne__(self, value):
        return md.not_equal(self, value)

    def __and__(self, value):
        return md.logical_and(self, value)

    def __or__(self, value):
        return md.logical_or(self, value)

    def __not__(self, value):
        return md.logical_not(self, value)

    def __xor__(self, value):
        return md.logical_xor(self, value)

    @property
    def __array_interface__(self):
        return self._data.__array_interface__

    def __array__(self, dtype=None, copy=None):
        if dtype is not None and dtype != self.dtype:
            if copy == False:
                raise ValueError("attempted cast, but copies are not permitted")
            return self._data.astype(dtype=dtype)
        if copy == True:
            return self._data.copy()
        return self._data


def ones_like(a: Tensor, allow_grad=False, **kwargs):
    return Tensor(np.ones_like(a._data, **kwargs), allow_grad=allow_grad)


def zeros_like(a: Tensor, allow_grad=False, **kwargs):
    return Tensor(np.zeros_like(a._data, **kwargs), allow_grad=allow_grad)


def full_like(a: Tensor, x, allow_grad=False, **kwargs):
    return Tensor(np.full_like(a._data, x, **kwargs), allow_grad=allow_grad)
