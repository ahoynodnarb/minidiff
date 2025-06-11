from builtins import all as py_all, any as py_any

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import minidiff as md
from .topology import FuncNode


class StatefulOp:
    def create_forward(self):
        raise NotImplementedError

    def create_grads(self):
        raise NotImplementedError


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
    # if the function is not differentiable, we still want to propagate the gradient to avoid breaking the
    # graph, but it is smarter to just zero out the gradients.
    if not is_differentiable:
        grad_funcs = [lambda a, b, grad: md.zeros_like(grad) for _ in grad_funcs]

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
            output = md.Tensor(output, allow_grad=allow_grad)

        # just in case
        output.allow_grad = allow_grad

        # only attach a node if we're allowed to track gradients right now, and the tensor wants to track its gradient
        if md.grad_allowed_() and allow_grad:
            # obviously tensors who don't want their gradients to be checked have no gradient function
            filtered_grad_funcs = [
                grad_func if grad_allowed else None
                for grad_func, grad_allowed in zip(grad_funcs, allowed_grads)
            ]

            func_node = FuncNode(
                output_tensor=output,
                input_tensors=[
                    md.Tensor(x) if not isinstance(x, md.Tensor) else x
                    for x in func_inputs
                ],
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
    casting="safe",
):
    return generate_arbitrary_op_func(
        forward_func=forward_func,
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
    return generate_arbitrary_op_func(
        forward_func=forward_func,
        grad_funcs=[grad_a, grad_b, grad_c],
        is_differentiable=is_differentiable,
        tensor_only=tensor_only,
        is_backend_op=is_backend_op,
        propagate_kwargs=propagate_kwargs,
        op_name=op_name,
        casting=casting,
    )


def generate_stateful_op_func(
    stateful_op,
    is_differentiable=True,
    tensor_only=False,
    is_backend_op=False,
    propagate_kwargs=False,
    op_name=None,
    casting="safe",
):
    instance = stateful_op()
    forward_func = instance.create_forward()
    grad_funcs = instance.create_grads()
    custom_op_name = f"{stateful_op.__name__}"
    return generate_arbitrary_op_func(
        forward_func=forward_func,
        grad_funcs=grad_funcs,
        is_differentiable=is_differentiable,
        tensor_only=tensor_only,
        is_backend_op=is_backend_op,
        propagate_kwargs=propagate_kwargs,
        op_name=custom_op_name if op_name is None else op_name,
        casting=casting,
    )


copy = generate_binary_op_func(
    forward_func=np.copy, grad_a=lambda a, grad: grad, is_backend_op=True, casting=None
)
s_ = generate_binary_op_func(
    forward_func=lambda a, key: a[int(key)],
    is_differentiable=False,
    is_backend_op=True,
    casting=None,
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
    casting=None,
)
reshape = generate_binary_op_func(
    forward_func=np.reshape,
    grad_a=lambda a, grad: grad.reshape(a.shape),
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
    forward_func=np.sum, grad_a=lambda a, grad: grad, is_backend_op=True, casting=None
)
mean = generate_unary_op_func(
    forward_func=np.mean,
    grad_a=lambda a, grad: grad / a.size,
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
absolute = generate_unary_op_func(
    forward_func=np.absolute, grad_a=lambda a, grad: grad * (a != 0), is_backend_op=True
)
all = generate_unary_op_func(
    forward_func=np.all, is_differentiable=False, is_backend_op=True, casting=None
)
any = generate_unary_op_func(
    forward_func=np.any, is_differentiable=False, is_backend_op=True, casting=None
)

__all__ = [
    "copy",
    "s_",
    "clip",
    "reshape",
    "matmul",
    "add",
    "subtract",
    "multiply",
    "true_divide",
    "floor_divide",
    "power",
    "sqrt",
    "floor",
    "ceil",
    "cos",
    "sin",
    "tan",
    "cosh",
    "sinh",
    "tanh",
    "exp",
    "log",
    "sum",
    "mean",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "absolute",
    "all",
    "any",
]
