try:
    import cupy as np  # type: ignore

    BACKEND = "cupy"
except ImportError:
    import numpy as np

    BACKEND = "numpy"

allow_grad = True


class no_grad:
    def __enter__(self):
        self.prev = grad_allowed()
        set_allow_grad(False)

    def __exit__(self, type, value, traceback):
        set_allow_grad(self.prev)


def set_allow_grad(allow):
    global allow_grad
    allow_grad = allow


def grad_allowed():
    global allow_grad
    return allow_grad


if __name__ == "__main__":
    with no_grad():
        print(grad_allowed())
    print(grad_allowed())
