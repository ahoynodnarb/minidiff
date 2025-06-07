# NEED TO MAKE SURE GRADIENTS ARE STABLE, LOOKING AT YOU LOG/EXP

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import minidiff
import topology

# compute from left to right, dy/dw2 then dw2/dw1 to get dy/dw1 and finally dw1/dx to get dy/dx
# dy/dw2 would just be the loss gradient

# all tensors by default should not allow grad
# everything created in no_grad maintains that no gradient state even when exiting unless updated
# everything created outside no_grad becomes no gradient within the scope, but reverts outside


class Tensor:
    def __init__(self, tensor, allow_grad=False, dtype=np.float32):
        if isinstance(tensor, np.ndarray):
            self._tensor = tensor
        else:
            self._tensor = np.array(tensor, dtype=dtype)

        self.grad = zeros_like(self, allow_grad=False) if allow_grad else None

        self.traversal_path = None
        self.func_node = None
        self.graphed = False
        self.allow_grad = allow_grad

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

    def backward(self, force_retraversal=False):
        if not self.allow_grad:
            return

        if self.func_node is None:
            return

        if force_retraversal or self.traversal_path is None:
            # it is usually faster just to keep this as a list because we're probably not
            # going to iterate over that many nodes, so a linear search is much faster and
            # much more worth the overhead than a set
            seen = []
            self.traversal_path = []

            # topologically sort
            def dfs(tensor):
                root = tensor.func_node
                if root is None or root in seen:
                    return
                seen.append(root)
                for input_tensor in root.input_tensors:
                    dfs(input_tensor)
                self.traversal_path.append(tensor)

            dfs(self)

        self.grad = ones_like(self, allow_grad=False)
        for tensor in reversed(self.traversal_path):
            n = tensor.func_node
            n_grad = tensor.grad
            n.update_grads(n_grad)

    def item(self):
        assert len(self) == 1
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
        assert isinstance(other, Tensor)
        return matmul(self, other)

    def __imatmul__(self, other):
        assert isinstance(other, Tensor)
        assert not self.allow_grad
        self._tensor = self._tensor @ other._tensor
        return self

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __iadd__(self, other):
        assert not self.allow_grad
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
        assert not self.allow_grad
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
        assert not self.allow_grad
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
        assert not self.allow_grad
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
        assert not self.allow_grad
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
        assert not self.allow_grad
        self._tensor = self._tensor**other

        return self

    def __neg__(self):
        ret = Tensor(-self._tensor, allow_grad=self.allow_grad)
        return ret

    def __repr__(self):
        return self._tensor.__repr__()

    def __len__(self):
        return self._tensor.__len__()

    def __getitem__(self, key):
        return Tensor(self._tensor[key], allow_grad=self.allow_grad)

    def __setitem__(self, key, val):
        assert not self.allow_grad
        assert not self.graphed
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


def _generate_unary_op_func(backend_func, grad_a=None, differentiable=True):
    if not differentiable:
        grad_a = lambda a, b, grad: 0

    def minidiff_func(a: Tensor, **kwargs):
        assert isinstance(a, Tensor)

        can_allow_grad = minidiff.grad_allowed() and a.allow_grad
        output = Tensor(backend_func(a._tensor, **kwargs), allow_grad=can_allow_grad)

        if can_allow_grad:
            func_node = topology.UnaryNode(a, grad_a)
            output.func_node = func_node
            output.graphed = True

        return output

    return minidiff_func


def _generate_binary_op_func(
    backend_func, grad_a=None, grad_b=None, differentiable=True, tensor_only=False
):
    if not differentiable:
        grad_a = lambda a, b, grad: 0
        grad_b = lambda a, b, grad: 0

    if tensor_only:

        def minidiff_func(a, b, **kwargs):
            assert isinstance(a, Tensor)
            assert isinstance(b, Tensor)

            allow_grad = a.allow_grad or b.allow_grad
            can_allow_grad = minidiff.grad_allowed() and allow_grad
            output = Tensor(
                backend_func(a._tensor, b._tensor, **kwargs), allow_grad=allow_grad
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
            assert not (a_is_scalar and b_is_scalar)

            allow_grad = (a_is_scalar or a.allow_grad) or (b_is_scalar or b.allow_grad)
            can_track_grad = minidiff.grad_allowed() and allow_grad
            if a_is_scalar:
                output = Tensor(
                    backend_func(a, b._tensor, **kwargs), allow_grad=allow_grad
                )
            elif b_is_scalar:
                output = Tensor(
                    backend_func(a._tensor, b, **kwargs), allow_grad=allow_grad
                )
            else:
                output = Tensor(
                    backend_func(a._tensor, b._tensor, **kwargs),
                    allow_grad=allow_grad,
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
    assert isinstance(a, Tensor)
    assert isinstance(shape, tuple)
    can_allow_grad = minidiff.grad_allowed() and a.allow_grad

    original_shape = a.shape
    output = Tensor(np.reshape(a._tensor, shape, **kwargs), allow_grad=can_allow_grad)

    if can_allow_grad:
        reshape_node = topology.UnaryNode(
            a, lambda a, grad: grad.reshape(original_shape)
        )
        output.func_node = reshape_node

    return output


matmul = _generate_binary_op_func(
    backend_func=np.matmul,
    grad_a=lambda a, b, grad: matmul(grad, b.t),
    grad_b=lambda a, b, grad: matmul(a.t, grad),
    tensor_only=True,
)
add = _generate_binary_op_func(
    backend_func=np.add, grad_a=lambda a, b, grad: grad, grad_b=lambda a, b, grad: grad
)
subtract = _generate_binary_op_func(
    backend_func=np.subtract,
    grad_a=lambda a, b, grad: grad,
    grad_b=lambda a, b, grad: -grad,
)
multiply = _generate_binary_op_func(
    backend_func=np.multiply,
    grad_a=lambda a, b, grad: grad * b,
    grad_b=lambda a, b, grad: grad * a,
)
true_divide = _generate_binary_op_func(
    backend_func=np.true_divide,
    grad_a=lambda a, b, grad: grad / b,
    grad_b=lambda a, b, grad: (-grad * a) / (b**2),
)
floor_divide = _generate_binary_op_func(
    backend_func=np.floor_divide, differentiable=False
)
power = _generate_binary_op_func(
    backend_func=np.power,
    grad_a=lambda a, b, grad: grad * b * (a ** (b - 1)),
    grad_b=lambda a, b, grad: grad * np.log(a) * a**b,
)


def sqrt(x, **kwargs):
    return power(x, 0.5, **kwargs)


floor = _generate_unary_op_func(backend_func=np.floor, differentiable=False)
ceil = _generate_unary_op_func(backend_func=np.ceil, differentiable=False)
cos = _generate_unary_op_func(
    backend_func=np.cos, grad_a=lambda a, grad: grad * -sin(a)
)
sin = _generate_unary_op_func(backend_func=np.sin, grad_a=lambda a, grad: grad * cos(a))
tan = _generate_unary_op_func(
    backend_func=np.tan, grad_a=lambda a, grad: grad * (1 / cos(a) ** 2)
)
cosh = _generate_unary_op_func(
    backend_func=np.cosh, grad_a=lambda a, grad: grad * sinh(a)
)
sinh = _generate_unary_op_func(
    backend_func=np.sinh, grad_a=lambda a, grad: grad * cosh(a)
)
tanh = _generate_unary_op_func(
    backend_func=np.sinh, grad_a=lambda a, grad: grad * (1 / cosh(a) ** 2)
)
exp = _generate_unary_op_func(backend_func=np.exp, grad_a=lambda a, grad: grad * exp(a))
log = _generate_unary_op_func(backend_func=np.log, grad_a=lambda a, grad: grad / a)
sum = _generate_unary_op_func(backend_func=np.log, grad_a=lambda a, grad: grad * sum(a))
mean = _generate_unary_op_func(
    backend_func=np.log, grad_a=lambda a, grad: grad * sum(a) / a.size
)
