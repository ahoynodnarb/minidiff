from builtins import all as py_all, any as py_any

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

from .topology import FuncNode
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
            self._data = data
        else:
            if dtype is None:
                self._data = np.array(data)
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
            self._data = np.add(self._data, other._data, casting="safe")
        else:
            self._data = np.add(self._data, other, casting="safe")

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
            self._data = np.subtract(self._data, other._data, casting="safe")
        else:
            self._data = np.subtract(self._data, other, casting="safe")

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
            self._data = np.multiply(self._data, other._data, casting="safe")
        else:
            self._data = np.multiply(self._data, other, casting="safe")

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
        return -1 * self

    def __repr__(self):
        return self._data.__repr__()

    def __len__(self):
        return self._data.__len__()

    def __getitem__(self, key):
        # this should be an op actually
        return s_(self, key)

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
        return greater(self, value)

    def __ge__(self, value):
        return greater_equal(self, value)

    def __lt__(self, value):
        return less(self, value)

    def __le__(self, value):
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
    return Tensor(np.ones_like(a._data, dtype=a.dtype, **kwargs), allow_grad=allow_grad)


def zeros_like(a: Tensor, allow_grad=False, **kwargs):
    return Tensor(
        np.zeros_like(a._data, dtype=a.dtype, **kwargs), allow_grad=allow_grad
    )


def full_like(a: Tensor, x, allow_grad=False, **kwargs):
    return Tensor(
        np.full_like(a._data, x, dtype=a.dtype, **kwargs), allow_grad=allow_grad
    )


def generate_arbitrary_op_func(
    forward_func,
    grad_funcs,
    is_differentiable=True,
    tensor_only=False,
    is_backend_op=False,
    propagate_kwargs=False,
    op_name=None,
    casting="safe",
):
    # if the function is not is_differentiable, we still want to propagate the gradient to avoid breaking the
    # graph, but it is smarter to just zero out the gradients.
    if not is_differentiable:
        grad_funcs = [lambda a, b, grad: zeros_like(grad) for _ in grad_funcs]

    # maybe I should split this into multiple functions without if statements since those are probably clunky and generate overhead
    def minidiff_func(*inputs, **kwargs):
        # no leafs can ever be created by an op
        inputs_are_tensors = [isinstance(x, Tensor) for x in inputs]

        if not tensor_only and not py_any(inputs_are_tensors):
            raise ValueError(
                "minidiff functions only work when at least one argument is a minidiff Tensor"
            )

        if tensor_only and not py_all(inputs_are_tensors):
            raise ValueError("This function only supports minidiff Tensors")

        as_tensors = [Tensor(x) if not isinstance(x, Tensor) else x for x in inputs]

        # allow gradient if at least one of the input tensors allows a gradient
        allowed_grads = [x.allow_grad for x in as_tensors]
        allow_grad = py_any(allowed_grads)

        forward_args = [x._data for x in as_tensors] if is_backend_op else as_tensors

        if casting is None:
            output = forward_func(*forward_args, **kwargs)
        else:
            output = forward_func(*forward_args, casting=casting, **kwargs)

        if is_backend_op:
            output = Tensor(output, allow_grad=allow_grad)

        # just in case
        output.allow_grad = allow_grad

        if grad_allowed_() and allow_grad:
            filtered_grad_funcs = [
                grad_func if grad_allowed else None
                for grad_func, grad_allowed in zip(grad_funcs, allowed_grads)
            ]

            func_node = FuncNode(
                output_tensor=output,
                input_tensors=as_tensors,
                grad_functions=filtered_grad_funcs,
            )
            func_node.op_name = forward_func.__name__ if op_name is None else op_name
            if propagate_kwargs:
                func_node.kwargs = kwargs

        return output

    return minidiff_func


def generate_unary_op_func(
    forward_func,
    grad_a=None,
    is_differentiable=True,
    is_backend_op=False,
    propagate_kwargs=False,
    op_name=None,
    casting=None,
):
    return generate_arbitrary_op_func(
        forward_func,
        grad_funcs=[grad_a],
        is_differentiable=is_differentiable,
        tensor_only=True,
        is_backend_op=is_backend_op,
        propagate_kwargs=propagate_kwargs,
        op_name=op_name,
        casting=casting,
    )


def generate_binary_op_func(
    forward_func,
    grad_a=None,
    grad_b=None,
    is_differentiable=True,
    tensor_only=False,
    is_backend_op=False,
    propagate_kwargs=False,
    op_name=None,
    casting="safe",
):
    return generate_arbitrary_op_func(
        forward_func,
        grad_funcs=[grad_a, grad_b],
        is_differentiable=is_differentiable,
        tensor_only=tensor_only,
        is_backend_op=is_backend_op,
        propagate_kwargs=propagate_kwargs,
        op_name=op_name,
        casting=casting,
    )


s_ = generate_binary_op_func(
    forward_func=lambda a, key, **kwargs: a[int(key)],
    is_differentiable=False,
    is_backend_op=True,
)
clip = generate_unary_op_func(
    forward_func=np.clip,
    grad_a=lambda a, grad, a_min=None, a_max=None: grad
    * logical_and(
        a > float("-inf") if a_min is None else a_min,
        a < float("inf") if a_max is None else a_max,
    ),
    propagate_kwargs=True,
    is_backend_op=True,
)
reshape = generate_unary_op_func(
    forward_func=np.reshape,
    grad_a=lambda a, grad: grad.reshape(a.shape),
    is_backend_op=True,
)
matmul = generate_binary_op_func(
    forward_func=np.matmul,
    grad_a=lambda a, b, grad: matmul(grad, b.t),
    grad_b=lambda a, b, grad: matmul(a.t, grad),
    tensor_only=True,
    is_backend_op=True,
)
add = generate_binary_op_func(
    forward_func=np.add,
    grad_a=lambda a, b, grad: grad,
    grad_b=lambda a, b, grad: grad,
    is_backend_op=True,
)
subtract = generate_binary_op_func(
    forward_func=np.subtract,
    grad_a=lambda a, b, grad: grad,
    grad_b=lambda a, b, grad: -grad,
    is_backend_op=True,
)
multiply = generate_binary_op_func(
    forward_func=np.multiply,
    grad_a=lambda a, b, grad: grad * b,
    grad_b=lambda a, b, grad: grad * a,
    is_backend_op=True,
)
true_divide = generate_binary_op_func(
    forward_func=np.true_divide,
    grad_a=lambda a, b, grad: grad / b,
    grad_b=lambda a, b, grad: (-grad * a) / (b**2),
    is_backend_op=True,
)
floor_divide = generate_binary_op_func(
    forward_func=np.floor_divide, is_differentiable=False, is_backend_op=True
)
power = generate_binary_op_func(
    forward_func=np.power,
    grad_a=lambda a, b, grad: grad * b * (a ** (b - 1)),
    grad_b=lambda a, b, grad: grad * log(a) * a**b,
    is_backend_op=True,
)
sqrt = lambda a, b, **kwargs: power(a, b, **kwargs)
floor = generate_unary_op_func(
    forward_func=np.floor, is_differentiable=False, is_backend_op=True
)
ceil = generate_unary_op_func(
    forward_func=np.ceil, is_differentiable=False, is_backend_op=True
)
cos = generate_unary_op_func(
    forward_func=np.cos, grad_a=lambda a, grad: grad * -sin(a), is_backend_op=True
)
sin = generate_unary_op_func(
    forward_func=np.sin, grad_a=lambda a, grad: grad * cos(a), is_backend_op=True
)
tan = generate_unary_op_func(
    forward_func=np.tan,
    grad_a=lambda a, grad: grad * (1 / cos(a) ** 2),
    is_backend_op=True,
)
cosh = generate_unary_op_func(
    forward_func=np.cosh, grad_a=lambda a, grad: grad * sinh(a), is_backend_op=True
)
sinh = generate_unary_op_func(
    forward_func=np.sinh, grad_a=lambda a, grad: grad * cosh(a), is_backend_op=True
)
tanh = generate_unary_op_func(
    forward_func=np.sinh,
    grad_a=lambda a, grad: grad * (1 / cosh(a) ** 2),
    is_backend_op=True,
)
exp = generate_unary_op_func(
    forward_func=np.exp, grad_a=lambda a, grad: grad * exp(a), is_backend_op=True
)
log = generate_unary_op_func(
    forward_func=np.log, grad_a=lambda a, grad: grad / a, is_backend_op=True
)
sum = generate_unary_op_func(
    forward_func=np.sum, grad_a=lambda a, grad: grad, is_backend_op=True
)
mean = generate_unary_op_func(
    forward_func=np.mean, grad_a=lambda a, grad: grad / a.size, is_backend_op=True
)
greater = generate_binary_op_func(
    forward_func=np.greater, is_differentiable=False, is_backend_op=True
)
greater_equal = generate_binary_op_func(
    forward_func=np.greater_equal, is_differentiable=False, is_backend_op=True
)
less = generate_binary_op_func(
    forward_func=np.less, is_differentiable=False, is_backend_op=True
)
less_equal = generate_binary_op_func(
    forward_func=np.less_equal, is_differentiable=False, is_backend_op=True
)
equal = generate_binary_op_func(
    forward_func=np.equal, is_differentiable=False, is_backend_op=True
)
not_equal = generate_binary_op_func(
    forward_func=np.not_equal, is_differentiable=False, is_backend_op=True
)
logical_and = generate_binary_op_func(
    forward_func=np.logical_and, is_differentiable=False, is_backend_op=True
)
logical_or = generate_binary_op_func(
    forward_func=np.logical_or, is_differentiable=False, is_backend_op=True
)
logical_not = generate_binary_op_func(
    forward_func=np.logical_not, is_differentiable=False, is_backend_op=True
)
logical_xor = generate_binary_op_func(
    forward_func=np.logical_xor, is_differentiable=False, is_backend_op=True
)
absolute = generate_unary_op_func(
    forward_func=np.absolute, grad_a=lambda a, grad: grad * (a != 0), is_backend_op=True
)
all = generate_unary_op_func(
    forward_func=np.all, is_differentiable=False, is_backend_op=True
)
any = generate_unary_op_func(
    forward_func=np.any, is_differentiable=False, is_backend_op=True
)
