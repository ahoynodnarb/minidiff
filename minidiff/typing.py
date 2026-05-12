from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, TypeVar, Union

    import minidiff as md

    BackendTensor = TypeVar("BackendTensor")

    TensorLike = Union[int, float, md.Tensor]

    dtype = TypeVar("dtype")

    GenericFunc = Callable[..., md.Tensor]
    GenericOp = GenericFunc
    GenericOpGrad = Callable[..., md.Tensor]

    UnaryFunc = Callable[[md.Tensor], md.Tensor]
    UnaryOp = UnaryFunc
    UnaryOpGrad = Callable[[md.Tensor, md.Tensor], md.Tensor]

    BinaryFunc = Union[
        Callable[[md.Tensor, Any], md.Tensor],
        Callable[[Any, md.Tensor], md.Tensor],
        Callable[[md.Tensor, md.Tensor], md.Tensor],
    ]
    BinaryOp = BinaryFunc
    BinaryOpGrad = Union[
        Callable[[md.Tensor, Any, md.Tensor], md.Tensor],
        Callable[[Any, md.Tensor, md.Tensor], md.Tensor],
        Callable[[md.Tensor, md.Tensor, md.Tensor], md.Tensor],
    ]

    TernaryFunc = Union[
        Callable[[md.Tensor, Any, Any], md.Tensor],
        Callable[[md.Tensor, md.Tensor, Any], md.Tensor],
        Callable[[md.Tensor, Any, md.Tensor], md.Tensor],
        Callable[[Any, md.Tensor, Any], md.Tensor],
        Callable[[Any, md.Tensor, md.Tensor], md.Tensor],
        Callable[[Any, Any, md.Tensor], md.Tensor],
        Callable[[md.Tensor, md.Tensor, md.Tensor], md.Tensor],
    ]
    TernaryOp = TernaryFunc
    TernaryOpGrad = Union[
        Callable[[md.Tensor, Any, Any, md.Tensor], md.Tensor],
        Callable[[md.Tensor, md.Tensor, Any, md.Tensor], md.Tensor],
        Callable[[md.Tensor, Any, md.Tensor, md.Tensor], md.Tensor],
        Callable[[Any, md.Tensor, Any, md.Tensor], md.Tensor],
        Callable[[Any, md.Tensor, md.Tensor, md.Tensor], md.Tensor],
        Callable[[Any, Any, md.Tensor, md.Tensor], md.Tensor],
        Callable[[md.Tensor, md.Tensor, md.Tensor, md.Tensor], md.Tensor],
    ]
