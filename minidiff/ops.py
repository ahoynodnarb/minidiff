from builtins import all as py_all, any as py_any

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import minidiff as md
from .topology import FuncNode


class Op:
    def create_forward(self):
        raise NotImplementedError

    def create_grads(self):
        raise NotImplementedError


def stateless_op_func(**kwargs):
    def wrapper(func):
        return generate_stateless_op_func(forward_func=func, **kwargs)

    return wrapper


def unary_op_func(**kwargs):
    def wrapper(func):
        return generate_unary_op_func(forward_func=func, **kwargs)

    return wrapper


def binary_op_func(**kwargs):
    def wrapper(func):
        return generate_binary_op_func(forward_func=func, **kwargs)

    return wrapper


def generate_op_func(
    op_type,
    is_differentiable=True,
    tensor_only=False,
    is_backend_op=False,
    propagate_kwargs=False,
    op_name=None,
    casting="safe",
):
    instance = op_type()
    forward_func = instance.create_forward()
    grad_funcs = instance.create_grads()
    # if the function is not differentiable, we still want to propagate the gradient to avoid breaking the
    # graph, but it is smarter to just zero out the gradients.
    if not is_differentiable:
        grad_funcs = [
            lambda a, b, grad: md.zeros_like(grad) for _ in range(len(grad_funcs))
        ]

    def minidiff_func(*func_inputs, **kwargs):
        tensor_inputs = [isinstance(x, md.Tensor) for x in func_inputs]

        if not tensor_only and not py_any(tensor_inputs):
            raise ValueError(
                "minidiff functions only work when at least one argument is a minidiff Tensor"
            )

        if tensor_only and not py_all(tensor_inputs):
            raise ValueError("This function only supports minidiff Tensors")

        # allow gradient if at least one of the input tensors allows a gradient
        allowed_grads = [
            x.allow_grad if isinstance(x, md.Tensor) else False for x in func_inputs
        ]
        allow_grad = py_any(allowed_grads)

        # if the op is a traditional numpy function, then we need to "uncast" it back to numpy
        if is_backend_op:
            forward_inputs = [
                x._data if isinstance(x, md.Tensor) else x for x in func_inputs
            ]
        else:
            forward_inputs = func_inputs

        if casting is None or not is_backend_op:
            output = forward_func(*forward_inputs, **kwargs)
        else:
            output = forward_func(*forward_inputs, casting=casting, **kwargs)

        # traditional numpy functions of course return numpy objects, so we need to wrap in a Tensor
        if is_backend_op:
            output = md.Tensor(output)

        # ensure gradient tracking rules do not break
        output.allow_grad = allow_grad

        # only attach a node if we're allowed to track gradients right now, and the tensor wants to track its gradient
        if md.grad_allowed_() and allow_grad:
            # obviously tensors who don't want their gradients to be checked have no gradient function
            filtered_grad_funcs = [
                grad_func if grad_allowed else None
                for grad_func, grad_allowed in zip(grad_funcs, allowed_grads)
            ]

            # FuncNodes can only track tensors, so we have to make everything a tensor.
            func_node = FuncNode(
                op_output=output,
                op_inputs=func_inputs,
                grad_functions=filtered_grad_funcs,
            )
            func_node.op_name = forward_func.__name__ if op_name is None else op_name
            if propagate_kwargs:
                func_node.kwargs = kwargs

        return output

    return minidiff_func


def generate_stateless_op_func(
    forward_func,
    grad_funcs,
    is_differentiable=True,
    tensor_only=False,
    is_backend_op=False,
    propagate_kwargs=False,
    op_name=None,
    casting="safe",
):
    class StatelessOp(Op):
        def create_forward(self):
            return forward_func

        def create_grads(self):
            return grad_funcs

    return generate_op_func(
        op_type=StatelessOp,
        is_differentiable=is_differentiable,
        tensor_only=tensor_only,
        is_backend_op=is_backend_op,
        propagate_kwargs=propagate_kwargs,
        op_name=op_name,
        casting=casting,
    )


def generate_unary_op_func(
    forward_func,
    grad=None,
    is_differentiable=True,
    is_backend_op=False,
    propagate_kwargs=False,
    op_name=None,
    casting="safe",
):
    return generate_stateless_op_func(
        forward_func=forward_func,
        grad_funcs=[grad],
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
    return generate_stateless_op_func(
        forward_func=forward_func,
        grad_funcs=[grad_a, grad_b],
        is_differentiable=is_differentiable,
        tensor_only=tensor_only,
        is_backend_op=is_backend_op,
        propagate_kwargs=propagate_kwargs,
        op_name=op_name,
        casting=casting,
    )


def generate_ternary_op_func(
    forward_func,
    grad_a=None,
    grad_b=None,
    grad_c=None,
    is_differentiable=True,
    tensor_only=False,
    is_backend_op=False,
    propagate_kwargs=False,
    op_name=None,
    casting="safe",
):
    return generate_stateless_op_func(
        forward_func=forward_func,
        grad_funcs=[grad_a, grad_b, grad_c],
        is_differentiable=is_differentiable,
        tensor_only=tensor_only,
        is_backend_op=is_backend_op,
        propagate_kwargs=propagate_kwargs,
        op_name=op_name,
        casting=casting,
    )


transpose = generate_binary_op_func(
    forward_func=np.transpose,
    grad_a=lambda a, grad, axes=None: transpose(grad, axes=axes),
    is_backend_op=True,
    propagate_kwargs=True,
    casting=None,
)
swapaxes = generate_ternary_op_func(
    forward_func=np.swapaxes,
    grad_a=lambda a, axis1, axis2, grad, **kwargs: swapaxes(
        grad, axis1, axis2, **kwargs
    ),
    is_backend_op=True,
    propagate_kwargs=True,
    casting=None,
)
flip = generate_unary_op_func(
    forward_func=np.flip,
    grad=lambda a, grad, **kwargs: flip(grad, **kwargs),
    is_backend_op=True,
    propagate_kwargs=True,
    casting=None,
)
broadcast_to = generate_binary_op_func(
    forward_func=np.broadcast_to,
    grad_a=lambda a, grad: md.collect_gradients(grad=grad, target_shape=a.shape),
    is_backend_op=True,
    casting=None,
)
atleast_1d = generate_unary_op_func(
    forward_func=np.atleast_1d,
    grad=lambda a, grad: grad,
    is_backend_op=True,
    casting=None,
)
atleast_2d = generate_unary_op_func(
    forward_func=np.atleast_2d,
    grad=lambda a, grad: grad,
    is_backend_op=True,
    casting=None,
)
atleast_3d = generate_unary_op_func(
    forward_func=np.atleast_3d,
    grad=lambda a, grad: grad,
    is_backend_op=True,
    casting=None,
)
copy = generate_binary_op_func(
    forward_func=np.copy, grad_a=lambda a, grad: grad, is_backend_op=True, casting=None
)


def s__grad(a, key, grad):
    ret = md.zeros_like(a)
    np.add.at(ret._data, key, grad._data)
    return ret


s_ = generate_binary_op_func(
    forward_func=lambda a, key: a[key],
    grad_a=s__grad,
    grad_b=None,
    is_backend_op=True,
    casting=None,
    op_name="index",
)
clip = generate_unary_op_func(
    forward_func=np.clip,
    grad=lambda a, grad, a_min=None, a_max=None: grad
    * logical_and(
        a > float("-inf") if a_min is None else a_min,
        a < float("inf") if a_max is None else a_max,
    ),
    propagate_kwargs=True,
    is_backend_op=True,
    casting=None,
)
reshape = generate_binary_op_func(
    forward_func=np.reshape,
    grad_a=lambda a, b, grad: grad.reshape(a.shape),
    grad_b=None,
    is_backend_op=True,
    casting=None,
)
matmul = generate_binary_op_func(
    forward_func=np.matmul,
    grad_a=lambda a, b, grad: matmul(grad, b.t),
    grad_b=lambda a, b, grad: matmul(a.t, grad),
    tensor_only=True,
    is_backend_op=True,
    casting=None,
)


def grad_a(a, b, grad, axes=2):
    if isinstance(axes, int):
        axes_a = tuple(range(a.ndim - axes, a.ndim))
        axes_b = tuple(range(axes))
        axes = (axes_a, axes_b)
    # indices of all dims in b not originally contracted in the forward tensordot
    uncontracted_a = tuple(i for i in range(a.ndim) if i not in axes[0])
    uncontracted_b = tuple(i for i in range(b.ndim) if i not in axes[1])
    # indices of all dims in grad that align with uncontracted_b
    grad_aligned = tuple(range(grad.ndim - len(uncontracted_b), grad.ndim))
    new_axes = (grad_aligned, uncontracted_b)
    result = tensordot(grad, b, axes=new_axes)
    # first few indices will be uncontracted in a, last few will be contracted in a (original forward pass)
    # need to transpose such that the first few take up the uncontracted a indices, and the last few take up the contracted a indices
    permutation_indices = [0] * a.ndim
    n_uncontracted_a = len(uncontracted_a)
    uncontracted_idx = 0
    contracted_idx = 0
    for i in range(a.ndim):
        if i < n_uncontracted_a:
            permutation_indices[uncontracted_a[uncontracted_idx]] = i
            uncontracted_idx += 1
        else:
            permutation_indices[axes[0][contracted_idx]] = i
            contracted_idx += 1

    reshaped = md.transpose(result, permutation_indices)
    return reshaped


def grad_b(a, b, grad, axes=2):
    if isinstance(axes, int):
        axes_a = tuple(range(a.ndim - axes, a.ndim))
        axes_b = tuple(range(axes))
        axes = (axes_a, axes_b)
    # indices of all dims in a not originally contracted in the forward tensordot
    uncontracted_a = tuple(i for i in range(a.ndim) if i not in axes[0])
    uncontracted_b = tuple(i for i in range(b.ndim) if i not in axes[1])
    # indices of all dims in grad that align with uncontracted_a
    grad_aligned = tuple(range(len(uncontracted_a)))
    new_axes = (uncontracted_a, grad_aligned)
    result = tensordot(a, grad, axes=new_axes)
    # first few indices of result will be contracted in a, last few will be uncontracted in b (original forward pass
    # need to transpose so that the last few take up the uncontracted b indices, and the first few take up the original contracted indices
    n_contracted_a = len(axes[0])
    contracted_idx = 0
    uncontracted_idx = 0
    permutation_indices = [0] * b.ndim
    for i in range(b.ndim):
        if i < n_contracted_a:
            permutation_indices[axes[1][contracted_idx]] = i
            contracted_idx += 1
        else:
            permutation_indices[uncontracted_b[uncontracted_idx]] = i
            uncontracted_idx += 1
    reshaped = md.transpose(result, permutation_indices)
    return reshaped


tensordot = generate_binary_op_func(
    forward_func=np.tensordot,
    grad_a=grad_a,
    grad_b=grad_b,
    tensor_only=True,
    is_backend_op=True,
    propagate_kwargs=True,
    casting=None,
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
sqrt = lambda a, b, **kwargs: power(a, 0.5, **kwargs)
floor = generate_unary_op_func(
    forward_func=np.floor, is_differentiable=False, is_backend_op=True
)
ceil = generate_unary_op_func(
    forward_func=np.ceil, is_differentiable=False, is_backend_op=True
)
cos = generate_unary_op_func(
    forward_func=np.cos, grad=lambda a, grad: grad * -sin(a), is_backend_op=True
)
sin = generate_unary_op_func(
    forward_func=np.sin, grad=lambda a, grad: grad * cos(a), is_backend_op=True
)
tan = generate_unary_op_func(
    forward_func=np.tan,
    grad=lambda a, grad: grad * (1 / cos(a) ** 2),
    is_backend_op=True,
)
cosh = generate_unary_op_func(
    forward_func=np.cosh, grad=lambda a, grad: grad * sinh(a), is_backend_op=True
)
sinh = generate_unary_op_func(
    forward_func=np.sinh, grad=lambda a, grad: grad * cosh(a), is_backend_op=True
)
tanh = generate_unary_op_func(
    forward_func=np.tanh,
    grad=lambda a, grad: grad * (1 / cosh(a) ** 2),
    is_backend_op=True,
)
exp = generate_unary_op_func(
    forward_func=np.exp, grad=lambda a, grad: grad * exp(a), is_backend_op=True
)
log = generate_unary_op_func(
    forward_func=np.log, grad=lambda a, grad: grad / a, is_backend_op=True
)
sum = generate_unary_op_func(
    forward_func=np.sum, grad=lambda a, grad: grad, is_backend_op=True, casting=None
)
mean = generate_unary_op_func(
    forward_func=np.mean,
    grad=lambda a, grad: grad / a.size,
    is_backend_op=True,
    casting=None,
)
greater = generate_binary_op_func(
    forward_func=np.greater, is_differentiable=False, is_backend_op=True, casting=None
)
greater_equal = generate_binary_op_func(
    forward_func=np.greater_equal,
    is_differentiable=False,
    is_backend_op=True,
    casting=None,
)
less = generate_binary_op_func(
    forward_func=np.less, is_differentiable=False, is_backend_op=True, casting=None
)
less_equal = generate_binary_op_func(
    forward_func=np.less_equal,
    is_differentiable=False,
    is_backend_op=True,
    casting=None,
)
equal = generate_binary_op_func(
    forward_func=np.equal, is_differentiable=False, is_backend_op=True, casting=None
)
not_equal = generate_binary_op_func(
    forward_func=np.not_equal, is_differentiable=False, is_backend_op=True, casting=None
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
sign = generate_unary_op_func(
    forward_func=np.sign, is_differentiable=False, is_backend_op=True
)
absolute = generate_unary_op_func(
    forward_func=np.absolute, grad=lambda a, grad: grad * sign(a), is_backend_op=True
)
all = generate_unary_op_func(
    forward_func=np.all, is_differentiable=False, is_backend_op=True, casting=None
)
any = generate_unary_op_func(
    forward_func=np.any, is_differentiable=False, is_backend_op=True, casting=None
)
