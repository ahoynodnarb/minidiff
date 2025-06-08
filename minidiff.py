try:
    import cupy as np  # type: ignore

    BACKEND = "cupy"
except ImportError:
    import numpy as np

    BACKEND = "numpy"

import topology
import contextvars

_allow_grad = contextvars.ContextVar("allow_grad", default=True)


class no_grad:
    def __enter__(self):
        self.prev = _allow_grad.get()
        _allow_grad.set(False)

    def __exit__(self, type, value, traceback):
        _allow_grad.set(self.prev)


def set_allow_grad(allow):
    _allow_grad.set(allow)


def grad_allowed():
    return _allow_grad.get()


# compute from left to right, dy/dw2 then dw2/dw1 to get dy/dw1 and finally dw1/dx to get dy/dx
# dy/dw2 would just be the loss gradient

# all tensors by default should not allow grad
# everything created in no_grad maintains that no gradient state even when exiting unless updated
# everything created outside no_grad becomes no gradient within the scope, but reverts outside


class Tensor:
    def __init__(self, tensor, allow_grad=False, dtype=np.float32, is_leaf=True):
        if isinstance(tensor, np.ndarray):
            self._tensor = tensor
        else:
            self._tensor = np.array(tensor, dtype=dtype)

        # tensors not created by ops are leafs. this property is immutable
        self._is_leaf = is_leaf
        self.func_node = None
        self.graphed = False
        self._allow_grad = allow_grad
        # don't store gradients unless we are user-created.
        self.grad = (
            zeros_like(self, allow_grad=False) if allow_grad and is_leaf else None
        )

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def allow_grad(self):
        return self._allow_grad

    @allow_grad.setter
    def allow_grad(self, allow_grad):
        assert not (
            not allow_grad and self.graphed
        ), "Tensors can only stop tracking gradients if they are not part of a computational graph"
        if self._allow_grad == allow_grad:
            return
        if not self.is_leaf or not allow_grad:
            self.grad = None
        else:
            self.grad = zeros_like(self, allow_grad=False)
        self._allow_grad = allow_grad

    @property
    def t(self):
        return self._tensor.t

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def size(self):
        return self._tensor.size

    @property
    def dtype(self):
        return self._tensor.dtype

    def backward(self, retain_graph=False):
        if not self.allow_grad:
            return

        if self.func_node is None:
            return

        # it is usually faster just to keep this as a list because we're probably not
        # going to iterate over that many nodes, so a linear search is much faster and
        # much more worth the overhead than a set
        seen = []
        traversal_path = []

        # topologically sort
        def dfs(tensor):
            root = tensor.func_node
            if root is None or root in seen:
                return
            # temporarily maintain gradients only during backward pass
            if not tensor.is_leaf:
                tensor.grad = zeros_like(tensor, allow_grad=False)
            seen.append(root)
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
        assert (
            len(self) == 1
        ), "only Tensors with a single element can be reduced to a Python scalar"
        return self._tensor[0].item()

    def reshape(self, shape, **kwargs):
        return reshape(self, shape, **kwargs)

    def dot(self, other, **kwargs):
        return matmul(self, other, **kwargs)

    def add(self, other, **kwargs):
        return add(self, other, **kwargs)

    def multiply(self, other, **kwargs):
        return multiply(self, other, **kwargs)

    def __matmul__(self, other):
        assert isinstance(other, Tensor), "can only matrix multiply between two Tensors"
        return matmul(self, other)

    def __imatmul__(self, other):
        assert isinstance(other, Tensor), "can only matrix multiply between two Tensors"
        assert (
            not self.allow_grad
        ), "in-place operations are not allowed while tracking gradients"
        self._tensor = self._tensor @ other._tensor
        return self

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __iadd__(self, other):
        assert (
            not self.allow_grad
        ), "in-place operations are not allowed while tracking gradients"
        if isinstance(other, Tensor):
            self._tensor += other._tensor
        else:
            self._tensor += self._tensor

        return self

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __isub__(self, other):
        assert (
            not self.allow_grad
        ), "in-place operations are not allowed while tracking gradients"
        if isinstance(other, Tensor):
            self._tensor = self._tensor - other._tensor
        else:
            self._tensor = self._tensor - other

        return self

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __imul__(self, other):
        assert (
            not self.allow_grad
        ), "in-place operations are not allowed while tracking gradients"
        if isinstance(other, Tensor):
            self._tensor = self._tensor * other._tensor
        else:
            self._tensor = self._tensor * other

        return self

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __itruediv__(self, other):
        assert (
            not self.allow_grad
        ), "in-place operations are not allowed while tracking gradients"
        if isinstance(other, Tensor):
            self._tensor = self._tensor / other._tensor
        else:
            self._tensor = self._tensor / other

        return self

    def __floordiv__(self, other):
        return floor_divide(self, other)

    def __rfloordiv__(self, other):
        return floor_divide(other, self)

    def __ifloordiv__(self, other):
        assert (
            not self.allow_grad
        ), "in-place operations are not allowed while tracking gradients"
        if isinstance(other, Tensor):
            self._tensor = self._tensor // other._tensor
        else:
            self._tensor = self._tensor // other

        return self

    def __pow__(self, other):
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __ipow__(self, other):
        assert (
            not self.allow_grad
        ), "in-place operations are not allowed while tracking gradients"
        self._tensor = self._tensor**other

        return self

    def __neg__(self):
        ret = Tensor(-self._tensor, allow_grad=self.allow_grad, is_leaf=self.is_leaf)
        return ret

    def __repr__(self):
        return self._tensor.__repr__()

    def __len__(self):
        return self._tensor.__len__()

    def __getitem__(self, key):
        return Tensor(
            self._tensor[key], allow_grad=self.allow_grad, is_leaf=self.is_leaf
        )

    def __setitem__(self, key, val):
        assert (
            not self.allow_grad
        ), "in-place operations are not allowed while tracking gradients"
        assert (
            not self.graphed
        ), "mutating tensors is not allowed if the tensor is on a computational graph"
        self._tensor[key] = val


def ones_like(a: Tensor, allow_grad=False, **kwargs):
    return Tensor(
        np.ones_like(a._tensor, dtype=a.dtype, **kwargs), allow_grad=allow_grad
    )


def zeros_like(a: Tensor, allow_grad=False, **kwargs):
    return Tensor(
        np.zeros_like(a._tensor, dtype=a.dtype, **kwargs), allow_grad=allow_grad
    )


def full_like(a: Tensor, x, allow_grad=False, **kwargs):
    return Tensor(
        np.full_like(a._tensor, x, dtype=a.dtype, **kwargs), allow_grad=allow_grad
    )


def _generate_unary_op_func(forward_func, grad_a=None, differentiable=True, backend_op=False):
    if not differentiable:
        grad_a = lambda a, b, grad: 0

    def minidiff_func(a: Tensor, **kwargs):
        assert isinstance(a, Tensor)

        can_allow_grad = grad_allowed() and a.allow_grad
        output = Tensor(
            forward_func(a._tensor if backend_op else a, **kwargs), allow_grad=a.allow_grad, is_leaf=False
        )

        if can_allow_grad:
            func_node = topology.UnaryNode(a, grad_a)
            output.func_node = func_node
            output.graphed = True

        return output

    return minidiff_func


def _generate_binary_op_func(
    forward_func, grad_a=None, grad_b=None, differentiable=True, tensor_only=False, backend_op=False
):
    if not differentiable:
        grad_a = lambda a, b, grad: 0
        grad_b = lambda a, b, grad: 0

    if tensor_only:

        def minidiff_func(a, b, **kwargs):
            assert isinstance(a, Tensor), "this function only supports minidiff Tensors"
            assert isinstance(b, Tensor), "this function only supports minidiff Tensors"

            allow_grad = a.allow_grad or b.allow_grad
            can_allow_grad = grad_allowed() and allow_grad
            output = Tensor(
                forward_func(a._tensor if backend_op else a, b._tensor if backend_op else a, **kwargs),
                allow_grad=allow_grad,
                is_leaf=False,
            )

            if can_allow_grad:
                func_node = topology.BinaryNode(
                    a,
                    b,
                    None if not a.allow_grad else grad_a,
                    None if not b.allow_grad else grad_b,
                )
                output.func_node = func_node
                output.graphed = True

            return output

    else:

        def minidiff_func(a, b, **kwargs):
            a_is_scalar = not isinstance(a, Tensor)
            b_is_scalar = not isinstance(b, Tensor)
            assert not (
                a_is_scalar and b_is_scalar
            ), "minidiff functions only work when at least one argument is a minidiff Tensor"

            allow_grad = (a_is_scalar or a.allow_grad) or (b_is_scalar or b.allow_grad)
            can_track_grad = grad_allowed() and allow_grad
            if not backend_op:
                output = Tensor(
                    forward_func(a, b, **kwargs),
                    allow_grad=allow_grad,
                    is_leaf=False,
                )
            elif a_is_scalar:
                output = Tensor(
                    forward_func(a, b._tensor, **kwargs),
                    allow_grad=allow_grad,
                    is_leaf=False,
                )
            elif b_is_scalar:
                output = Tensor(
                    forward_func(a._tensor, b, **kwargs),
                    allow_grad=allow_grad,
                    is_leaf=False,
                )
            else:
                output = Tensor(
                    forward_func(a._tensor, b._tensor, **kwargs),
                    allow_grad=allow_grad,
                    is_leaf=False,
                )

            if can_track_grad:
                func_node = topology.BinaryNode(
                    a,
                    b,
                    None if a_is_scalar or not a.allow_grad else grad_a,
                    None if b_is_scalar or not b.allow_grad else grad_b,
                )
                output.func_node = func_node
                output.graphed = True

            return output

    return minidiff_func


# default behavior to allow grad if a allows it unless otherwise specified
def reshape(a: Tensor, shape: tuple, **kwargs):
    assert isinstance(a, Tensor), "input argument is not a Tensor"
    assert isinstance(shape, tuple), "shape argument is not a Tuple"
    can_allow_grad = grad_allowed() and a.allow_grad

    original_shape = a.shape
    output = Tensor(
        np.reshape(a._tensor, shape, **kwargs), allow_grad=a.allow_grad, is_leaf=False
    )

    if can_allow_grad:
        reshape_node = topology.UnaryNode(
            a, lambda a, grad: grad.reshape(original_shape)
        )
        output.func_node = reshape_node

    return output


matmul = _generate_binary_op_func(
    forward_func=np.matmul,
    grad_a=lambda a, b, grad: matmul(grad, b.t),
    grad_b=lambda a, b, grad: matmul(a.t, grad),
    tensor_only=True,
    backend_op=True
)
add = _generate_binary_op_func(
    forward_func=np.add, grad_a=lambda a, b, grad: grad, grad_b=lambda a, b, grad: grad, backend_op=True
)
subtract = _generate_binary_op_func(
    forward_func=np.subtract,
    grad_a=lambda a, b, grad: grad,
    grad_b=lambda a, b, grad: -grad,
    backend_op=True
)
multiply = _generate_binary_op_func(
    forward_func=np.multiply,
    grad_a=lambda a, b, grad: grad * b,
    grad_b=lambda a, b, grad: grad * a,
    backend_op=True
)
true_divide = _generate_binary_op_func(
    forward_func=np.true_divide,
    grad_a=lambda a, b, grad: grad / b,
    grad_b=lambda a, b, grad: (-grad * a) / (b**2),
    backend_op=True
)
floor_divide = _generate_binary_op_func(
    forward_func=np.floor_divide, differentiable=False, backend_op=True
)
power = _generate_binary_op_func(
    forward_func=np.power,
    grad_a=lambda a, b, grad: grad * b * (a ** (b - 1)),
    grad_b=lambda a, b, grad: grad * np.log(a) * a**b,
    backend_op=True
)


def sqrt(x, **kwargs):
    return power(x, 0.5, **kwargs)


floor = _generate_unary_op_func(forward_func=np.floor, differentiable=False, backend_op=True)
ceil = _generate_unary_op_func(forward_func=np.ceil, differentiable=False, backend_op=True)
cos = _generate_unary_op_func(
    forward_func=np.cos, grad_a=lambda a, grad: grad * -sin(a), backend_op=True
)
sin = _generate_unary_op_func(forward_func=np.sin, grad_a=lambda a, grad: grad * cos(a), backend_op=True)
tan = _generate_unary_op_func(
    forward_func=np.tan, grad_a=lambda a, grad: grad * (1 / cos(a) ** 2), backend_op=True
)
cosh = _generate_unary_op_func(
    forward_func=np.cosh, grad_a=lambda a, grad: grad * sinh(a), backend_op=True
)
sinh = _generate_unary_op_func(
    forward_func=np.sinh, grad_a=lambda a, grad: grad * cosh(a), backend_op=True
)
tanh = _generate_unary_op_func(
    forward_func=np.sinh, grad_a=lambda a, grad: grad * (1 / cosh(a) ** 2), backend_op=True
)
exp = _generate_unary_op_func(forward_func=np.exp, grad_a=lambda a, grad: grad * exp(a), backend_op=True)
log = _generate_unary_op_func(forward_func=np.log, grad_a=lambda a, grad: grad / a, backend_op=True)
sum = _generate_unary_op_func(forward_func=np.log, grad_a=lambda a, grad: grad, backend_op=True)
mean = _generate_unary_op_func(
    forward_func=np.log, grad_a=lambda a, grad: grad / a.size, backend_op=True
)
