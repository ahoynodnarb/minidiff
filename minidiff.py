from numbers import Number

try:
    import cupy as np  # type: ignore

    BACKEND = "cupy"
except ImportError:
    import numpy as np

    BACKEND = "numpy"

DEBUG = True

import topology
import contextvars

_allow_grad = contextvars.ContextVar("allow_grad", default=True)
_allow_leafs = contextvars.ContextVar("allow_leafs", default=True)


class no_grad:
    def __enter__(self):
        self.prev = _allow_grad.get()
        set_allow_grad(False)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev)


class no_leafs:
    def __enter__(self):
        self.prev = _allow_leafs.get()
        set_allow_leafs(False)

    def __exit__(self, type, value, traceback):
        set_allow_leafs(self.prev)


def set_allow_grad(allow):
    _allow_grad.set(allow)


def grad_allowed():
    return _allow_grad.get()


def set_allow_leafs(allow):
    _allow_leafs.set(allow)


def leafs_allowed():
    return _allow_leafs.get()


# compute from left to right, dy/dw2 then dw2/dw1 to get dy/dw1 and finally dw1/dx to get dy/dx
# dy/dw2 would just be the loss gradient

# all tensors by default should not allow grad
# all tensors keep their allow_grad state whether in no_grad() or not, no_grad() just prevents any graph creation


class Tensor:
    def __init__(self, data, allow_grad=False, dtype=np.float32, is_leaf=True):
        if isinstance(data, np.ndarray):
            self._data = data
        else:
            if dtype is None:
                self._data = np.array(data)
            else:
                self._data = np.array(data, dtype=dtype)

        self._allow_grad = allow_grad
        # tensors not created by ops are leafs. this property is immutable
        self._func_node = None
        self._is_leaf = isinstance(data, Number) or (is_leaf and leafs_allowed())
        # don't store gradients unless we are user-created.
        self.graphed = False
        self.grad = (
            zeros_like(self, allow_grad=False) if allow_grad and is_leaf else None
        )

    @property
    def func_node(self):
        return self._func_node

    @func_node.setter
    def func_node(self, func_node):
        if self.is_leaf and func_node is not None:
            raise ValueError("leaf tensors cannot possess func_nodes")

        self._func_node = func_node

    @property
    def is_leaf(self):
        return self._is_leaf

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

        if not self.is_leaf or not allow_grad:
            self.grad = None
        else:
            self.grad = zeros_like(self, allow_grad=False)

        self._allow_grad = allow_grad

    @property
    def t(self):
        return self._data.t

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    def backward(self, retain_graph=False):
        if not self.allow_grad:
            return

        if self.func_node is None:
            return

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

        self.grad = ones_like(self, allow_grad=False)

        for tensor in reversed(traversal_path):
            n = tensor.func_node
            n_grad = tensor.grad
            n.update_grads(n_grad)
            # we're only temporarily storing grads, so we need to remove any references when
            # we're done for the sake of memory
            if not tensor.is_leaf:
                tensor.grad = None
            if not retain_graph:
                tensor.func_node = None

    def item(self):
        if self.size != 1:
            raise ValueError(
                "only Tensors with a single element can be reduced to a Python scalar"
            )

        return self._data.item()

    def clip(self, a_min=None, a_max=None):
        return clip(self, a_min=a_min, a_max=a_max)

    def reshape(self, shape, **kwargs):
        return reshape(self, shape, **kwargs)

    def dot(self, other, **kwargs):
        return matmul(self, other, **kwargs)

    def add(self, other, **kwargs):
        return add(self, other, **kwargs)

    def multiply(self, other, **kwargs):
        return multiply(self, other, **kwargs)

    def __matmul__(self, other):
        return matmul(self, other)

    def __imatmul__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        self._data = self._data @ other._data
        return self

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __iadd__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data += other._data
        else:
            self._data += other

        return self

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __isub__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = self._data - other._data
        else:
            self._data = self._data - other

        return self

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __imul__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = self._data * other._data
        else:
            self._data = self._data * other

        return self

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __itruediv__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = self._data / other._data
        else:
            self._data = self._data / other

        return self

    def __floordiv__(self, other):
        return floor_divide(self, other)

    def __rfloordiv__(self, other):
        return floor_divide(other, self)

    def __ifloordiv__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        if isinstance(other, Tensor):
            self._data = self._data // other._data
        else:
            self._data = self._data // other

        return self

    def __pow__(self, other):
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __ipow__(self, other):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )

        self._data = self._data**other

        return self

    def __neg__(self):
        # this should be an op actually
        ret = Tensor(-self._data, allow_grad=self.allow_grad, is_leaf=False)
        return ret

    def __repr__(self):
        return self._data.__repr__()

    def __len__(self):
        return self._data.__len__()

    def __getitem__(self, key):
        # this should be an op actually
        return Tensor(self._data[key], allow_grad=self.allow_grad, is_leaf=False)

    def __setitem__(self, key, val):
        if self.allow_grad:
            raise ValueError(
                "in-place operations are not allowed while tracking gradients"
            )
        if self.graphed:
            raise ValueError(
                "mutating tensors is not allowed if the tensor is on a computational graph"
            )

        self._data[key] = val

    def __gt__(self, value):
        return greater(self, value)

    def __gte__(self, value):
        return greater_equal(self, value)

    def __lt__(self, value):
        return less(self, value)

    def __lte__(self, value):
        return less_equal(self, value)

    def __eq__(self, value):
        return equal(self, value)

    def __ne__(self, value):
        return not equal(self, value)

    def __and__(self, value):
        return logical_and(self, value)

    def __or__(self, value):
        return logical_or(self, value)

    def __not__(self, value):
        return logical_not(self, value)

    def __xor__(self, value):
        return logical_xor(self, value)


def ones_like(a: Tensor, allow_grad=False, **kwargs):
    return Tensor(
        np.ones_like(a._data, dtype=a.dtype, **kwargs), allow_grad=allow_grad
    )


def zeros_like(a: Tensor, allow_grad=False, **kwargs):
    return Tensor(
        np.zeros_like(a._data, dtype=a.dtype, **kwargs), allow_grad=allow_grad
    )


def full_like(a: Tensor, x, allow_grad=False, **kwargs):
    return Tensor(
        np.full_like(a._data, x, dtype=a.dtype, **kwargs), allow_grad=allow_grad
    )


def _generate_unary_op_func(
    forward_func,
    grad_a=None,
    differentiable=True,
    backend_op=False,
    propagate_kwargs=False,
    op_name=None,
):
    if not differentiable:
        grad_a = lambda a, b, grad: zeros_like(grad)

    def minidiff_func(a: Tensor, **kwargs):
        if not isinstance(a, Tensor):
            raise ValueError("This function only supports minidiff Tensors")

        with no_leafs():
            can_allow_grad = grad_allowed() and a.allow_grad

            if backend_op:
                output = Tensor(
                    forward_func(a._data, **kwargs),
                    allow_grad=a.allow_grad,
                    is_leaf=False,
                )
            else:
                with no_grad(), no_leafs():
                    output = forward_func(a, **kwargs)

            if can_allow_grad:
                func_node = topology.UnaryNode(a, grad_a)
                func_node.op_name = (
                    forward_func.__name__ if op_name is None else op_name
                )
                if propagate_kwargs:
                    func_node.kwargs = kwargs

                output.func_node = func_node
                output.graphed = True

        return output

    return minidiff_func


def _generate_binary_op_func(
    forward_func,
    grad_a=None,
    grad_b=None,
    differentiable=True,
    tensor_only=False,
    backend_op=False,
    propagate_kwargs=False,
    op_name=None,
):
    # if the function is not differentiable, we still want to propagate the gradient to avoid breaking the
    # graph, but it is smarter to just zero out the gradients.
    if not differentiable:
        grad_a = lambda a, b, grad: zeros_like(grad)
        grad_b = lambda a, b, grad: zeros_like(grad)

    # maybe I should split this into multiple functions without if statements since those are probably clunky and generate overhead
    def minidiff_func(a, b, **kwargs):
        # no leafs can ever be created by an op
        a_is_tensor = isinstance(a, Tensor)
        b_is_tensor = isinstance(b, Tensor)
        if tensor_only:
            if not (a_is_tensor and b_is_tensor):
                raise ValueError("This function only supports minidiff Tensors")
        else:
            if not (a_is_tensor or b_is_tensor):
                raise ValueError(
                    "minidiff functions only work when at least one argument is a minidiff Tensor"
                )

        with no_leafs():

            if not a_is_tensor:
                a = Tensor(a)
            if not b_is_tensor:
                b = Tensor(b)

            # allow gradient if at least one of the input tensors allows a gradient
            allow_grad = a.allow_grad or b.allow_grad
            can_track_grad = grad_allowed() and allow_grad

            if backend_op:
                output = Tensor(
                    forward_func(a._data, b._data, **kwargs), allow_grad=allow_grad
                )
            else:
                with no_grad():
                    output = forward_func(a, b, **kwargs)

            if can_track_grad:
                first_grad = grad_a if a.allow_grad else None
                second_grad = grad_b if b.allow_grad else None

                func_node = topology.BinaryNode(a, b, first_grad, second_grad)
                func_node.op_name = (
                    forward_func.__name__ if op_name is None else op_name
                )
                if propagate_kwargs:
                    func_node.kwargs = kwargs

                output.func_node = func_node
                output.graphed = True

        return output

    return minidiff_func


clip = _generate_unary_op_func(
    forward_func=np.clip,
    grad_a=lambda a, grad, a_min=None, a_max=None: grad
    * Tensor(np.asarray(a_min < a._data < a_max)),
    propagate_kwargs=True,
    backend_op=True,
)
# should technically be unary, but I like to preserve interoperability so it will take in two inputs instead of a shape kwarg
reshape = _generate_unary_op_func(
    forward_func=np.reshape,
    grad_a=lambda a, grad: grad.reshape(a.shape),
    backend_op=True,
)

matmul = _generate_binary_op_func(
    forward_func=np.matmul,
    grad_a=lambda a, b, grad: matmul(grad, b.t),
    grad_b=lambda a, b, grad: matmul(a.t, grad),
    tensor_only=True,
    backend_op=True,
)
add = _generate_binary_op_func(
    forward_func=np.add,
    grad_a=lambda a, b, grad: grad,
    grad_b=lambda a, b, grad: grad,
    backend_op=True,
)
subtract = _generate_binary_op_func(
    forward_func=np.subtract,
    grad_a=lambda a, b, grad: grad,
    grad_b=lambda a, b, grad: -grad,
    backend_op=True,
)
multiply = _generate_binary_op_func(
    forward_func=np.multiply,
    grad_a=lambda a, b, grad: grad * b,
    grad_b=lambda a, b, grad: grad * a,
    backend_op=True,
)
true_divide = _generate_binary_op_func(
    forward_func=np.true_divide,
    grad_a=lambda a, b, grad: grad / b,
    grad_b=lambda a, b, grad: (-grad * a) / (b**2),
    backend_op=True,
)
floor_divide = _generate_binary_op_func(
    forward_func=np.floor_divide, differentiable=False, backend_op=True
)
power = _generate_binary_op_func(
    forward_func=np.power,
    grad_a=lambda a, b, grad: grad * b * (a**(b - 1)),
    grad_b=lambda a, b, grad: grad * np.log(a) * a**b,
    backend_op=True,
)
sqrt = lambda a, b, **kwargs: power(a, b, **kwargs)
floor = _generate_unary_op_func(
    forward_func=np.floor, differentiable=False, backend_op=True
)
ceil = _generate_unary_op_func(
    forward_func=np.ceil, differentiable=False, backend_op=True
)
cos = _generate_unary_op_func(
    forward_func=np.cos, grad_a=lambda a, grad: grad * -sin(a), backend_op=True
)
sin = _generate_unary_op_func(
    forward_func=np.sin, grad_a=lambda a, grad: grad * cos(a), backend_op=True
)
tan = _generate_unary_op_func(
    forward_func=np.tan,
    grad_a=lambda a, grad: grad * (1 / cos(a)**2),
    backend_op=True,
)
cosh = _generate_unary_op_func(
    forward_func=np.cosh, grad_a=lambda a, grad: grad * sinh(a), backend_op=True
)
sinh = _generate_unary_op_func(
    forward_func=np.sinh, grad_a=lambda a, grad: grad * cosh(a), backend_op=True
)
tanh = _generate_unary_op_func(
    forward_func=np.sinh,
    grad_a=lambda a, grad: grad * (1 / cosh(a)**2),
    backend_op=True,
)
exp = _generate_unary_op_func(
    forward_func=np.exp, grad_a=lambda a, grad: grad * exp(a), backend_op=True
)
log = _generate_unary_op_func(
    forward_func=np.log, grad_a=lambda a, grad: grad / a, backend_op=True
)
sum = _generate_unary_op_func(
    forward_func=np.sum, grad_a=lambda a, grad: grad, backend_op=True
)
mean = _generate_unary_op_func(
    forward_func=np.mean, grad_a=lambda a, grad: grad / a.size, backend_op=True
)
greater = _generate_binary_op_func(
    forward_func=np.greater, differentiable=False, backend_op=True
)
greater_equal = _generate_binary_op_func(
    forward_func=np.greater_equal, differentiable=False, backend_op=True
)
less = _generate_binary_op_func(
    forward_func=np.less, differentiable=False, backend_op=True
)
less_equal = _generate_binary_op_func(
    forward_func=np.less_equal, differentiable=False, backend_op=True
)
equal = _generate_binary_op_func(
    forward_func=np.equal, differentiable=False, backend_op=True
)
not_equal = _generate_binary_op_func(
    forward_func=np.not_equal, differentiable=False, backend_op=True
)
logical_and = _generate_binary_op_func(
    forward_func=np.logical_and, differentiable=False, backend_op=True
)
logical_or = _generate_binary_op_func(
    forward_func=np.logical_or, differentiable=False, backend_op=True
)
logical_not = _generate_binary_op_func(
    forward_func=np.logical_not, differentiable=False, backend_op=True
)
logical_xor = _generate_binary_op_func(
    forward_func=np.logical_xor, differentiable=False, backend_op=True
)
