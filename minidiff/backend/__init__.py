from __future__ import annotations

import importlib
from argparse import ArgumentParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

parser = ArgumentParser()
parser.add_argument(
    "--backend", help="specify selected backend", required=False, default=None
)
args = vars(parser.parse_args())

SPECIFIED_BACKEND = args["backend"]
DEFAULT_BACKENDS = [
    "minidiff.backend.cupy",
    "minidiff.backend.mlx",
    "minidiff.backend.numpy",
]


def import_backend(backend_name: str, package_name: Optional[str] = None):
    # https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
    module = importlib.import_module(backend_name, package=package_name)
    module_dict = module.__dict__
    if "__all__" in module_dict:
        module_exports = module_dict["__all__"]
    else:
        module_exports = [x for x in module_dict if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in module_exports})


def attempt_backend_import():
    used_backend = None

    def attempt_import(possible_backend: Optional[str]) -> bool:
        if possible_backend is None:
            return False

        try:
            import_backend(possible_backend)
            return True
        except:
            return False

    if attempt_import(SPECIFIED_BACKEND):
        used_backend = SPECIFIED_BACKEND

    if used_backend is None:
        for possible_backend in DEFAULT_BACKENDS:
            if attempt_import(possible_backend):
                used_backend = possible_backend
                break

    if used_backend is None:
        raise Exception("could not find a suitable backend")

    if SPECIFIED_BACKEND is not None and SPECIFIED_BACKEND != used_backend:
        print(
            f"could not find backend named {SPECIFIED_BACKEND}, defaulting to {used_backend} instead"
        )
