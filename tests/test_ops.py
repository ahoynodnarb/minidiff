from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minidiff as md
import minidiff.backend as backend
from minidiff.utils import compute_grads, try_unwrap

if TYPE_CHECKING:
    from typing import Any, Dict, Optional, Sequence

    import minidiff.typing as mdt


def perform_test(
    func: mdt.GenericFunc,
    backend_func: mdt.GenericFunc,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    forward_rtol: float = 1e-05,
    forward_atol: float = 1e-08,
    backward_rtol: float = 1e-02,
    backward_atol: float = 1e-05,
    exclude: Optional[Sequence[md.Tensor]] = None,
):
    out = func(*args, **kwargs)._data
    comp = backend_func(
        *[try_unwrap(x) for x in args], **{k: try_unwrap(v) for k, v in kwargs.items()}
    )

    def loss_func(*loss_args):
        actual = func(*loss_args, **kwargs)
        expected = md.zeros_like(actual)
        return md.sum((expected - actual) ** 2) / 2

    if out.size != 1:
        forward_mask = backend.tensor_constructor(
            ~(np.isnan(np.array(out)) | np.isnan(np.array(comp)))
        )
        out = out * forward_mask
        comp = comp * forward_mask
    assert np.allclose(
        out, comp, rtol=forward_rtol, atol=forward_atol
    ), f"❌ Forward Test failed for {func}. Compared against {backend_func}\nminidiff:\n{out}\nnumpy:\n{comp}"

    manual_grads, auto_grads = compute_grads(
        *args, func=loss_func, exclude=exclude, h=1e-5
    )
    for i, (manual, auto) in enumerate(zip(manual_grads, auto_grads)):
        if manual is None and auto is None:
            continue
        if manual.size != 1:
            grad_mask = backend.tensor_constructor(
                ~(np.isnan(np.array(manual)) | np.isnan(np.array(auto)))
            )
            manual = manual * grad_mask
            auto = auto * grad_mask
        assert np.allclose(
            manual, auto, rtol=backward_rtol, atol=backward_atol
        ), f"❌ Gradient Test wrt {i}th parameter failed for {func}. \nmanual gradients:\n{manual}\nautomatic gradients:\n{auto}"


def test_ravel():
    for _ in range(5):
        perform_test(
            func=md.ravel,
            backend_func=backend.ravel,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_flatten():
    for _ in range(5):
        perform_test(
            func=md.flatten,
            backend_func=backend.flatten,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_squeeze():
    for _ in range(5):
        perform_test(
            func=md.squeeze,
            backend_func=backend.squeeze,
            args=[
                md.randn(1, 2, 1, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_expand_dims():
    for _ in range(5):
        perform_test(
            func=md.expand_dims,
            backend_func=backend.expand_dims,
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
            backend_func=backend.max,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axis": random.randint(0, 3)},
        )


def test_min():
    for _ in range(5):
        perform_test(
            func=md.min,
            backend_func=backend.min,
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
            backend_func=backend.where,
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
            backend_func=backend.prod,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axis": tuple(random.sample(range(4), k=random.randint(0, 4)))},
            backward_rtol=1e-01,
        )


def test_transpose():
    for _ in range(5):
        axes = md.permutation(md.arange(4))
        # print(axes)
        perform_test(
            func=md.transpose,
            backend_func=backend.transpose,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axes": axes},
        )


def test_swapaxes():
    for _ in range(5):
        perform_test(
            func=md.swapaxes,
            backend_func=backend.swapaxes,
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
            backend_func=backend.flip,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={"axis": random.randint(0, 3)},
        )


def test_dot():
    for _ in range(5):
        perform_test(
            func=md.dot,
            backend_func=backend.dot,
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
            backend_func=backend.broadcast_to,
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
            backend_func=backend.atleast_1d,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_atleast_2d():
    for _ in range(5):
        perform_test(
            func=md.atleast_2d,
            backend_func=backend.atleast_2d,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_atleast_3d():
    for _ in range(5):
        perform_test(
            func=md.atleast_3d,
            backend_func=backend.atleast_3d,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
        )


def test_copy():
    for _ in range(5):
        perform_test(
            func=md.copy,
            backend_func=backend.copy,
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
            backend_func=backend.clip,
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
            backend_func=backend.reshape,
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
            backend_func=backend.matmul,
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
            backend_func=backend.tensordot,
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
            backend_func=backend.add,
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
            backend_func=backend.subtract,
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
            backend_func=backend.multiply,
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
            backend_func=backend.true_divide,
            args=[
                md.randn(2, 2, 2, 2, allow_grad=True),
                md.randn(2, 2, 2, 2, allow_grad=True),
            ],
            kwargs={},
            backward_rtol=1e-01,
            backward_atol=1e-05,
        )


def test_power():
    for _ in range(5):
        perform_test(
            func=md.power,
            backend_func=backend.power,
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
            backend_func=backend.cos,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
            backward_rtol=1e-01,
            backward_atol=1e-04,
        )


def test_sin():
    for _ in range(5):
        perform_test(
            func=md.sin,
            backend_func=backend.sin,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
            backward_rtol=1e-01,
            backward_atol=1e-04,
        )


def test_tan():
    for _ in range(5):
        perform_test(
            func=md.tan,
            backend_func=backend.tan,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
            backward_rtol=1e-01,
            backward_atol=1e-04,
        )


def test_cosh():
    for _ in range(5):
        perform_test(
            func=md.cosh,
            backend_func=backend.cosh,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
            backward_rtol=1e-01,
            backward_atol=1e-04,
        )


def test_sinh():
    for _ in range(5):
        perform_test(
            func=md.sinh,
            backend_func=backend.sinh,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
            backward_rtol=1e-01,
            backward_atol=1e-04,
        )


def test_tanh():
    for _ in range(5):
        perform_test(
            func=md.tanh,
            backend_func=backend.tanh,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
            backward_rtol=1e-01,
            backward_atol=1e-04,
        )


def test_exp():
    for _ in range(5):
        perform_test(
            func=md.exp,
            backend_func=backend.exp,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_log():
    for _ in range(5):
        perform_test(
            func=md.log,
            backend_func=backend.log,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_sum():
    for _ in range(5):
        perform_test(
            func=md.sum,
            backend_func=backend.sum,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_mean():
    for _ in range(5):
        perform_test(
            func=md.mean,
            backend_func=backend.mean,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


def test_absolute():
    for _ in range(5):
        perform_test(
            func=md.absolute,
            backend_func=backend.absolute,
            args=[md.randn(2, 2, 2, 2, allow_grad=True)],
            kwargs={},
        )


if __name__ == "__main__":
    test_ravel()
    test_flatten()
    test_squeeze()
    test_expand_dims()
    test_max()
    test_min()
    test_where()
    test_prod()
    test_transpose()
    test_swapaxes()
    test_flip()
    test_dot()
    test_broadcast_to()
    test_atleast_1d()
    test_atleast_2d()
    test_atleast_3d()
    test_copy()
    test_getitem()
    test_clip()
    test_reshape()
    test_matmul()
    test_tensordot()
    test_add()
    test_subtract()
    test_multiply()
    test_true_divide()
    test_power()
    test_cos()
    test_sin()
    test_tan()
    test_cosh()
    test_sinh()
    test_tanh()
    test_exp()
    test_log()
    test_sum()
    test_mean()
    test_absolute()
