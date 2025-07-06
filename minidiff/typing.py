from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, TypeVar, Union

    import minidiff as md

    T = TypeVar("T")

    TensorLike = Union[int, float, md.Tensor]

    dtype = Union[
        md.float64,
        md.float32,
        md.float16,
        md.uint64,
        md.uint32,
        md.uint16,
        md.uint8,
        md.int64,
        md.int32,
        md.int16,
        md.int8,
        md.bool,
    ]

    GenericFunc = Callable[..., md.Tensor]
    GenericOp = GenericFunc
    GenericOpGrad = Callable[..., md.Tensor]
    # class GenericFunc(Protocol):
    #     def __call__(self, *args: Any) -> md.Tensor: ...

    # class GenericOp(Protocol):
    #     def __call__(self, *args: Any) -> md.Tensor: ...

    # class GenericOpGrad(Protocol):
    #     def __call__(self, *args: md.Tensor) -> md.Tensor: ...

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
