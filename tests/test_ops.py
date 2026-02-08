from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minidiff as md
from minidiff.backend import current_backend
from minidiff.utils import compute_grads

if TYPE_CHECKING:
    from typing import Any, Dict, Optional, Sequence, Tuple

    import minidiff.typing as mdt


def filter_nan(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    condition = np.isnan(a) | np.isnan(b)
    a = np.where(condition, 0, a)
    b = np.where(condition, 0, b)
    return a, b


def perform_test(
    func: mdt.GenericFunc,
    backend_func: mdt.GenericFunc,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    forward_rtol: float = 1e-03,
    forward_atol: float = 1e-04,
    backward_rtol: float = 1e-03,
    backward_atol: float = 1e-04,
    exclude: Optional[Sequence[md.Tensor]] = None,
):
    out = func(*args, **kwargs)._data
    comp = backend_func(
        *[md.try_unwrap(x) for x in args],
        **{k: md.try_unwrap(v) for k, v in kwargs.items()},
    )

    def loss_func(*loss_args):
        actual = func(*loss_args, **kwargs)
        expected = md.zeros_like(actual)
        return md.sum((expected - actual) ** 2) / 2

    if out.size != 1:
        out, comp = filter_nan(
            current_backend.as_numpy(out), current_backend.as_numpy(comp)
        )
    assert np.allclose(
        out, comp, rtol=forward_rtol, atol=forward_atol
    ), f"❌ Forward Test failed for {func}. Compared against {backend_func}\nminidiff:\n{out}\nnumpy:\n{comp}"

    manual_grads, auto_grads = compute_grads(
        *args, func=loss_func, exclude=exclude, h=1e-2
    )
    for i, (manual, auto) in enumerate(zip(manual_grads, auto_grads)):
        if manual is None and auto is None:
            continue
        manual, auto = filter_nan(
            current_backend.as_numpy(manual), current_backend.as_numpy(auto)
        )
        # print(np.argwhere(np.abs((manual - auto) / manual) > 1e-1))
        assert np.allclose(
            manual, auto, rtol=backward_rtol, atol=backward_atol
        ), f"❌ Gradient Test wrt {i}th parameter failed for {func}. \nmanual gradients:\n{manual}\nautomatic gradients:\n{auto}\ndifference:\n{manual - auto}"
        # indices =
        # print()


def test_ravel():
    for _ in range(5):
        perform_test(
            func=md.ravel,
            backend_func=current_backend.ravel,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_flatten():
    for _ in range(5):
        perform_test(
            func=md.flatten,
            backend_func=current_backend.flatten,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_squeeze():
    for _ in range(5):
        perform_test(
            func=md.squeeze,
            backend_func=current_backend.squeeze,
            args=[
                md.randn(1, 2, 1, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_expand_dims():
    for _ in range(5):
        perform_test(
            func=md.expand_dims,
            backend_func=current_backend.expand_dims,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                tuple(random.sample(range(4), k=random.randint(0, 4))),
            ],
            kwargs={},
        )


def test_max():
    for _ in range(5):
        perform_test(
            func=md.max,
            backend_func=current_backend.max,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axis": random.randint(0, 3)},
        )


def test_min():
    for _ in range(5):
        perform_test(
            func=md.min,
            backend_func=current_backend.min,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axis": random.randint(0, 3)},
        )


def test_where():
    for _ in range(5):
        indices = md.binomial(1, random.uniform(0, 1), (2, 2, 2, 2))
        perform_test(
            func=md.where,
            backend_func=current_backend.where,
            args=[
                indices,
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
            exclude=[indices],
        )


def test_prod():
    for _ in range(5):
        perform_test(
            func=md.prod,
            backend_func=current_backend.prod,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axis": tuple(random.sample(range(4), k=random.randint(0, 4)))},
        )


def test_std():
    for _ in range(5):
        perform_test(
            func=md.std,
            backend_func=current_backend.std,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axis": tuple(random.sample(range(4), k=random.randint(0, 4)))},
        )


def test_transpose():
    for _ in range(5):
        axes = md.permutation(md.arange(4))
        perform_test(
            func=md.transpose,
            backend_func=current_backend.transpose,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axes": tuple(current_backend.as_numpy(axes._data))},
        )


def test_swapaxes():
    for _ in range(5):
        perform_test(
            func=md.swapaxes,
            backend_func=current_backend.swapaxes,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                random.randint(0, 3),
                random.randint(0, 3),
            ],
            kwargs={},
        )


def test_flip():
    for _ in range(5):
        perform_test(
            func=md.flip,
            backend_func=current_backend.flip,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axis": random.randint(0, 3)},
        )


def test_dot():
    for _ in range(5):
        perform_test(
            func=md.dot,
            backend_func=current_backend.dot,
            args=[
                md.randn(2, allow_grad=True),
                md.randn(2, allow_grad=True),
            ],
            kwargs={},
        )


def test_broadcast_to():
    for _ in range(5):
        perform_test(
            func=md.broadcast_to,
            backend_func=current_backend.broadcast_to,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                (4, 2, 2, 2, 2),
            ],
            kwargs={},
        )


def test_atleast_1d():
    for _ in range(5):
        perform_test(
            func=md.atleast_1d,
            backend_func=current_backend.atleast_1d,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_atleast_2d():
    for _ in range(5):
        perform_test(
            func=md.atleast_2d,
            backend_func=current_backend.atleast_2d,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_atleast_3d():
    for _ in range(5):
        perform_test(
            func=md.atleast_3d,
            backend_func=current_backend.atleast_3d,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_copy():
    for _ in range(5):
        perform_test(
            func=md.copy,
            backend_func=current_backend.copy,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_getitem():
    for _ in range(5):
        indices = md.randint(low=(0, 0, 0, 0), high=(2, 2, 2, 2))
        perform_test(
            func=md.getitem,
            backend_func=lambda x, key: x[key],
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                indices,
            ],
            kwargs={},
            exclude=[indices],
        )


def test_clip():
    for _ in range(5):
        perform_test(
            func=md.clip,
            backend_func=current_backend.clip,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                random.uniform(0, 10),
                random.uniform(-10, 0),
            ],
            kwargs={},
        )


def test_reshape():
    for _ in range(5):
        perform_test(
            func=md.reshape,
            backend_func=current_backend.reshape,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                (4, 4),
            ],
            kwargs={},
        )


def test_matmul():
    for _ in range(5):
        perform_test(
            func=md.matmul,
            backend_func=current_backend.matmul,
            args=[
                md.randn(10, 30, allow_grad=True),
                md.randn(30, 20, allow_grad=True),
            ],
            kwargs={},
        )


def test_tensordot():
    for _ in range(5):
        perform_test(
            func=md.tensordot,
            backend_func=current_backend.tensordot,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_add():
    for _ in range(5):
        perform_test(
            func=md.add,
            backend_func=current_backend.add,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_subtract():
    for _ in range(5):
        perform_test(
            func=md.subtract,
            backend_func=current_backend.subtract,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_multiply():
    for _ in range(5):
        perform_test(
            func=md.multiply,
            backend_func=current_backend.multiply,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_true_divide():
    for _ in range(5):
        perform_test(
            func=md.true_divide,
            backend_func=current_backend.true_divide,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_power():
    for _ in range(5):
        perform_test(
            func=md.power,
            backend_func=current_backend.power,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_cos():
    for _ in range(5):
        perform_test(
            func=md.cos,
            backend_func=current_backend.cos,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_sin():
    for _ in range(5):
        perform_test(
            func=md.sin,
            backend_func=current_backend.sin,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_tan():
    for _ in range(5):
        perform_test(
            func=md.tan,
            backend_func=current_backend.tan,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_cosh():
    for _ in range(5):
        perform_test(
            func=md.cosh,
            backend_func=current_backend.cosh,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_sinh():
    for _ in range(5):
        perform_test(
            func=md.sinh,
            backend_func=current_backend.sinh,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_tanh():
    for _ in range(5):
        perform_test(
            func=md.tanh,
            backend_func=current_backend.tanh,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_exp():
    for _ in range(5):
        perform_test(
            func=md.exp,
            backend_func=current_backend.exp,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_log():
    for _ in range(5):
        perform_test(
            func=md.log,
            backend_func=current_backend.log,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_sum():
    for _ in range(5):
        perform_test(
            func=md.sum,
            backend_func=current_backend.sum,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_mean():
    for _ in range(5):
        perform_test(
            func=md.mean,
            backend_func=current_backend.mean,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_absolute():
    for _ in range(5):
        perform_test(
            func=md.absolute,
            backend_func=current_backend.absolute,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )
