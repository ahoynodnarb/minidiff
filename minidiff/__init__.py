from contextvars import ContextVar

from .backend import _attempt_backend_import

_attempt_backend_import()

from .ops.definitions import *
from .tensor import *

_allow_grad = ContextVar("allow_grad", default=True)
_allow_new_grads = ContextVar("allow_new_grads", default=True)


class disable_new_grads:
    def __enter__(self):
        self.prev_allow_grad = _allow_grad.get()
        self.prev_allow_new_grads = _allow_new_grads.get()
        set_allow_grad(False)
        set_allow_new_grads(False)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev_allow_grad)
        set_allow_new_grads(self.prev_allow_new_grads)


class no_grad:
    def __enter__(self):
        self.prev = _allow_grad.get()
        set_allow_grad(False)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev)


class enable_grad:
    def __init__(self, enable: py_bool):
        self.enable = enable

    def __enter__(self):
        self.prev = _allow_grad.get()
        set_allow_grad(self.enable)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev)


def set_allow_new_grads(allow: py_bool):
    _allow_new_grads.set(allow)


def new_grads_allowed_() -> py_bool:
    return _allow_new_grads.get()


def set_allow_grad(allow: py_bool):
    _allow_grad.set(allow)


def grad_allowed_() -> py_bool:
    return _allow_grad.get()


dtype = current_backend.dtype
float64 = current_backend.float64
float32 = current_backend.float32
float16 = current_backend.float16
uint64 = current_backend.uint64
uint32 = current_backend.uint32
uint16 = current_backend.uint16
uint8 = current_backend.uint8
int64 = current_backend.int64
int32 = current_backend.int32
int16 = current_backend.int16
int8 = current_backend.int8
bool = current_backend.bool
nan = current_backend.nan
pi = 3.1415926535897932384626433
