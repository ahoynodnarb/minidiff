from typing import Any, Callable, Protocol, Sequence, TypeVar, Union

import minidiff as md

T = TypeVar("T")

NestedSequence = Union[Sequence[T], Sequence["NestedSequence[T]"]]

TensorLike = Union["md.Tensor", float, int]


class GenericFunc(Protocol):
    def __call__(self, *args: TensorLike) -> TensorLike: ...


class GenericOp(Protocol):
    def __call__(self, *args: Any) -> "md.Tensor": ...


class GenericOpGrad(Protocol):
    def __call__(self, *args: "md.Tensor") -> "md.Tensor": ...


UnaryFunc = Union[
    Callable[[Any], TensorLike],
    Callable[["md.Tensor"], "md.Tensor"],
]
UnaryOp = Callable[["md.Tensor"], "md.Tensor"]
UnaryOpGrad = Callable[["md.Tensor", "md.Tensor"], "md.Tensor"]

BinaryFunc = Union[
    Callable[[Any, Any], TensorLike],
    Callable[["md.Tensor", "md.Tensor"], "md.Tensor"],
]
BinaryOp = Union[
    Callable[[Any, Any], "md.Tensor"],
    Callable[["md.Tensor", "md.Tensor"], "md.Tensor"],
]
BinaryOpGrad = Callable[["md.Tensor", "md.Tensor", "md.Tensor"], "md.Tensor"]

TernaryFunc = Union[
    Callable[[Any, Any, Any], TensorLike],
    Callable[["md.Tensor", "md.Tensor", "md.Tensor"], "md.Tensor"],
]
TernaryOp = Union[
    Callable[[Any, Any, Any], "md.Tensor"],
    Callable[["md.Tensor", "md.Tensor", "md.Tensor"], "md.Tensor"],
]
TernaryOpGrad = Callable[
    ["md.Tensor", "md.Tensor", "md.Tensor", "md.Tensor"], "md.Tensor"
]
