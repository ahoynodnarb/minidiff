from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Protocol, Sequence, TypeVar, Union

    import minidiff as md

    T = TypeVar("T")

    NestedSequence = Union[Sequence[T], Sequence["NestedSequence[T]"]]

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

    class GenericFunc(Protocol):
        def __call__(self, *args: Any) -> Any: ...

    class GenericOp(Protocol):
        def __call__(self, *args: Any) -> md.Tensor: ...

    class GenericOpGrad(Protocol):
        def __call__(self, *args: md.Tensor) -> md.Tensor: ...

    UnaryFunc = Union[
        Callable[[Any], md.Tensor],
        Callable[[md.Tensor], md.Tensor],
    ]
    UnaryOp = Callable[[md.Tensor], md.Tensor]
    UnaryOpGrad = Callable[[md.Tensor, md.Tensor], md.Tensor]

    BinaryFunc = Union[
        Callable[[Any, Any], md.Tensor],
        Callable[[md.Tensor, md.Tensor], md.Tensor],
    ]
    BinaryOp = Union[
        Callable[[Any, Any], md.Tensor],
        Callable[[md.Tensor, md.Tensor], md.Tensor],
    ]
    BinaryOpGrad = Callable[[md.Tensor, md.Tensor, md.Tensor], md.Tensor]

    TernaryFunc = Union[
        Callable[[Any, Any, Any], md.Tensor],
        Callable[[md.Tensor, md.Tensor, md.Tensor], md.Tensor],
    ]
    TernaryOp = Union[
        Callable[[Any, Any, Any], md.Tensor],
        Callable[[md.Tensor, md.Tensor, md.Tensor], md.Tensor],
    ]
    TernaryOpGrad = Callable[[md.Tensor, md.Tensor, md.Tensor, md.Tensor], md.Tensor]
