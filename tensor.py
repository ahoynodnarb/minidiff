try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import minidiff
import topology

# compute from left to right, dy/dw2 then dw2/dw1 to get dy/dw1 and finally dw1/dx to get dy/dx
# dy/dw2 would just be the loss gradient


class Tensor:
    def __init__(self, tensor, allow_grad=True):
        if isinstance(tensor, np.ndarray):
            self._tensor = tensor
        else:
            self._tensor = np.array(tensor)

        self.grad = zeros_like(self, allow_grad=False) if allow_grad else None

        self.diff_node = None
        self._allow_grad = allow_grad

    @property
    def allow_grad(self):
        return self._allow_grad and minidiff.grad_allowed()

    @allow_grad.setter
    def allow_grad(self, allow_grad):
        self._allow_grad = allow_grad

    @property
    def t(self):
        return self._tensor.t

    @property
    def shape(self):
        return self._tensor.shape

    def backward(self):
        if not self.allow_grad:
            return

        if self.diff_node is None:
            return

        seen = []
        stack = []

        # topologically sort
        def dfs(tensor):
            root = tensor.diff_node
            if root is None or root in seen:
                return
            seen.append(root)
            for input_tensor in root.input_tensors:
                dfs(input_tensor)
            stack.append(tensor)

        dfs(self)
        self.grad = ones_like(self, allow_grad=False)
        for tensor in reversed(stack):
            n = tensor.diff_node
            n_grad = tensor.grad
            n.update_grads(n_grad)

    def reshape(self, *args, **kwargs):
        self._tensor = self._tensor.reshape(*args, **kwargs)

    def dot(self, other):
        return self @ other

    def add(self, other):
        return add(self, other)

    def multiply(self, other):
        return multiply(self, other)

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
        return truediv(self, other)

    def __rtruediv__(self, other):
        return truediv(other, self)

    def __itruediv__(self, other):
        assert not self.allow_grad
        if isinstance(other, Tensor):
            self._tensor = self._tensor / other._tensor
        else:
            self._tensor = self._tensor / other

        return self

    def __floordiv__(self, other):
        return floordiv(self, other)

    def __rfloordiv__(self, other):
        return floordiv(other, self)

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
        ret = Tensor(-self._tensor, allow_grad=self._allow_grad)
        return ret

    def __repr__(self):
        return self._tensor.__repr__()


def ones_like(t1: Tensor, allow_grad=True):
    return Tensor(np.ones_like(t1._tensor), allow_grad=allow_grad)


def zeros_like(t1: Tensor, allow_grad=True):
    return Tensor(np.zeros_like(t1._tensor), allow_grad=allow_grad)


def full_like(t1: Tensor, x, allow_grad=True):
    return Tensor(np.full_like(t1._tensor, x), allow_grad=allow_grad)


def _generate_unary_backend_interop(backend_func, grad_a=None, differentiable=True):
    def minidiff_func(a: Tensor, allow_grad=True, **kwargs):
        assert isinstance(a, Tensor)

        can_allow_grad = minidiff.grad_allowed() and allow_grad and differentiable
        output = Tensor(backend_func(a._tensor, **kwargs), allow_grad=can_allow_grad)

        if can_allow_grad:
            with minidiff.no_grad():
                func_node = topology.UnaryNode(a, grad_a)
                output.diff_node = func_node

        return output

    return minidiff_func


def _generate_binary_backend_interop(
    backend_func, grad_a=None, grad_b=None, differentiable=True, tensor_only=False
):
    if tensor_only:

        def minidiff_func(a, b, allow_grad=True, **kwargs):
            assert isinstance(a, Tensor)
            assert isinstance(b, Tensor)

            can_allow_grad = minidiff.grad_allowed() and allow_grad and differentiable
            output = Tensor(
                backend_func(a._tensor, b._tensor, **kwargs), allow_grad=can_allow_grad
            )

            if can_allow_grad:
                with minidiff.no_grad():
                    func_node = topology.BinaryNode(a, b, grad_a, grad_b)
                    output.diff_node = func_node

            return output

    else:

        def minidiff_func(a, b, allow_grad=True, **kwargs):
            a_is_scalar = not isinstance(a, Tensor)
            b_is_scalar = not isinstance(b, Tensor)
            assert not (a_is_scalar and b_is_scalar)

            can_allow_grad = minidiff.grad_allowed() and allow_grad and differentiable
            if a_is_scalar:
                output = Tensor(
                    backend_func(a, b._tensor, **kwargs), allow_grad=can_allow_grad
                )
            elif b_is_scalar:
                output = Tensor(
                    backend_func(a._tensor, b, **kwargs), allow_grad=can_allow_grad
                )
            else:
                output = Tensor(
                    backend_func(a._tensor, b._tensor, **kwargs),
                    allow_grad=can_allow_grad,
                )

            if can_allow_grad:
                with minidiff.no_grad():
                    func_node = topology.BinaryNode(
                        a,
                        b,
                        None if a_is_scalar else grad_a,
                        None if b_is_scalar else grad_b,
                    )
                    output.diff_node = func_node

            return output

    return minidiff_func


matmul = _generate_binary_backend_interop(
    backend_func=np.matmul,
    grad_a=lambda a, b, grad: grad.matmul(b.t),
    grad_b=lambda a, b, grad: a.t.matmul(grad),
    tensor_only=True,
)
add = _generate_binary_backend_interop(
    backend_func=np.add, grad_a=lambda a, b, grad: grad, grad_b=lambda a, b, grad: grad
)
subtract = _generate_binary_backend_interop(
    backend_func=np.subtract,
    grad_a=lambda a, b, grad: grad,
    grad_b=lambda a, b, grad: -grad,
)
multiply = _generate_binary_backend_interop(
    backend_func=np.multiply,
    grad_a=lambda a, b, grad: grad * b,
    grad_b=lambda a, b, grad: grad * a,
)
truediv = _generate_binary_backend_interop(
    backend_func=np.true_divide,
    grad_a=lambda a, b, grad: grad / b,
    grad_b=lambda a, b, grad: (-grad * a) / (b**2),
)
floordiv = _generate_binary_backend_interop(
    backend_func=np.floor_divide, differentiable=False
)
power = _generate_binary_backend_interop(
    backend_func=np.power,
    grad_a=lambda a, b, grad: grad * b * (a ** (b - 1)),
    grad_b=lambda a, b, grad: grad * np.log(a) * a**b,
)
floor = _generate_unary_backend_interop(backend_func=np.floor, differentiable=False)
cos = _generate_unary_backend_interop(
    backend_func=np.cos, grad_a=lambda a, grad: grad * -sin(a)
)
sin = _generate_unary_backend_interop(
    backend_func=np.sin, grad_a=lambda a, grad: grad * cos(a)
)
exp = _generate_unary_backend_interop(
    backend_func=np.exp, grad_a=lambda a, grad: grad * exp(a)
)
