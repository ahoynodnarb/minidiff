import numpy as np

import minidiff as md
from minidiff.utils import compute_grads


def perform_test(
    func,
    backend_func,
    args,
    kwargs,
    forward_rtol=1e-05,
    forward_atol=1e-08,
    backward_rtol=1e-05,
    backward_atol=1e-08,
):
    out = func(*args, **kwargs)._data
    comp = backend_func(*args, **kwargs)

    def loss_func(*loss_args):
        actual = func(*loss_args, **kwargs)
        expected = md.zeros_like(actual)
        return md.sum((expected - actual) ** 2) / 2

    mask = ~(np.isnan(out) | np.isnan(comp))
    assert np.allclose(
        out[mask], comp[mask], rtol=forward_rtol, atol=forward_atol
    ), f"❌ Forward Test failed for {func}. Compared against {backend_func}\nminidiff:\n{out}\nnumpy:\n{comp}"

    print(f"✅ Forward Test successful for {func}. Compared against {backend_func}")

    manual_grads, auto_grads = compute_grads(*args, func=loss_func)
    for i, (manual, auto) in enumerate(zip(manual_grads, auto_grads)):
        if manual is None or auto is None:
            continue
        mask = ~(np.isnan(manual) | np.isnan(auto))
        assert np.allclose(
            manual[mask], auto[mask], rtol=backward_rtol, atol=backward_atol
        ), f"❌ Gradient Test wrt {i}th parameter failed for {func}. \nmanual gradients:\n{manual}\nautomatic gradients:\n{auto}"

    print(f"✅ Gradient Test successful for {func}.")


def perform_tests(cases):
    for case in cases:
        func = case["func"]
        backend_func = case["backend_func"]
        args = case["args"]
        kwargs = case["kwargs"]
        if "forward_rtol" in case:
            forward_rtol = case["forward_rtol"]
        else:
            forward_rtol = 1e-04
        if "forward_atol" in case:
            forward_atol = case["forward_atol"]
        else:
            forward_atol = 1e-07
        if "backward_rtol" in case:
            backward_rtol = case["backward_rtol"]
        else:
            backward_rtol = 1e-04
        if "backward_atol" in case:
            backward_atol = case["backward_atol"]
        else:
            backward_atol = 1e-07
        perform_test(
            func,
            backend_func,
            args,
            kwargs,
            forward_rtol=forward_rtol,
            forward_atol=forward_atol,
            backward_rtol=backward_rtol,
            backward_atol=backward_atol,
        )


def test():
    import random

    cases = [
        {
            "func": md.ravel,
            "backend_func": lambda x: x.ravel(),
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.flatten,
            "backend_func": lambda x: x.flatten(),
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.squeeze,
            "backend_func": np.squeeze,
            "args": [
                md.randn(1, 2, 1, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.expand_dims,
            "backend_func": np.expand_dims,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                tuple(random.sample(range(4), k=random.randint(0, 4))),
            ],
            "kwargs": {},
        },
        {
            "func": md.max,
            "backend_func": np.max,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {"axis": random.randint(0, 3)},
        },
        {
            "func": md.where,
            "backend_func": np.where,
            "args": [
                md.binomial(1, random.uniform(0, 1), (2, 2, 2, 2)),
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.prod,
            "backend_func": np.prod,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {"axis": tuple(random.sample(range(4), k=random.randint(0, 4)))},
            "backward_atol": 1,
        },
        {
            "func": md.transpose,
            "backend_func": np.transpose,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {"axes": md.permutation(range(4))},
        },
        {
            "func": md.swapaxes,
            "backend_func": np.swapaxes,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                random.randint(0, 3),
                random.randint(0, 3),
            ],
            "kwargs": {},
        },
        {
            "func": md.flip,
            "backend_func": np.flip,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {"axis": random.randint(0, 3)},
        },
        {
            "func": md.dot,
            "backend_func": np.dot,
            "args": [
                md.randn(2, allow_grad=True),
                md.randn(2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.broadcast_to,
            "backend_func": np.broadcast_to,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                (4, 2, 2, 2, 2),
            ],
            "kwargs": {},
        },
        {
            "func": md.atleast_1d,
            "backend_func": np.atleast_1d,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.atleast_2d,
            "backend_func": np.atleast_2d,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.atleast_3d,
            "backend_func": np.atleast_3d,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.copy,
            "backend_func": np.copy,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.getitem,
            "backend_func": lambda x, key: x[key],
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randint(low=(0, 0, 0, 0), high=(2, 2, 2, 2)),
            ],
            "kwargs": {},
        },
        {
            "func": md.clip,
            "backend_func": np.clip,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                random.uniform(0, 10),
                random.uniform(-10, 0),
            ],
            "kwargs": {},
        },
        {
            "func": md.reshape,
            "backend_func": np.reshape,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                (4, 4),
            ],
            "kwargs": {},
        },
        {
            "func": md.matmul,
            "backend_func": np.matmul,
            "args": [
                md.randn(2, 2, allow_grad=True),
                md.randn(2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.tensordot,
            "backend_func": np.tensordot,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.add,
            "backend_func": np.add,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.subtract,
            "backend_func": np.subtract,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.multiply,
            "backend_func": np.multiply,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.true_divide,
            "backend_func": np.true_divide,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
            "backward_rtol": 1e-03,
            "backward_atol": 1e-05,
        },
        {
            "func": md.power,
            "backend_func": np.power,
            "args": [
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            "kwargs": {},
        },
        {
            "func": md.cos,
            "backend_func": np.cos,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
            "backward_rtol": 1e-03,
            "backward_atol": 1e-06,
        },
        {
            "func": md.sin,
            "backend_func": np.sin,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
            "backward_rtol": 1e-03,
            "backward_atol": 1e-06,
        },
        {
            "func": md.tan,
            "backend_func": np.tan,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
            "backward_rtol": 1e-03,
            "backward_atol": 1e-06,
        },
        {
            "func": md.cosh,
            "backend_func": np.cosh,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
            "backward_rtol": 1e-03,
            "backward_atol": 1e-06,
        },
        {
            "func": md.sinh,
            "backend_func": np.sinh,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
            "backward_rtol": 1e-03,
            "backward_atol": 1e-06,
        },
        {
            "func": md.tanh,
            "backend_func": np.tanh,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
            "backward_rtol": 1e-03,
            "backward_atol": 1e-06,
        },
        {
            "func": md.exp,
            "backend_func": np.exp,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
        },
        {
            "func": md.log,
            "backend_func": np.log,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
        },
        {
            "func": md.sum,
            "backend_func": np.sum,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
        },
        {
            "func": md.mean,
            "backend_func": np.mean,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
        },
        {
            "func": md.absolute,
            "backend_func": np.absolute,
            "args": [md.randn(2, 2, 2, 2, allow_grad=True)],
            "kwargs": {},
        },
    ]
    perform_tests(cases)
