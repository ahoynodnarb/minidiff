from __future__ import annotations

import importlib
from argparse import ArgumentParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

_parser = ArgumentParser()
_parser.add_argument(
    "--backend", help="specify selected backend", required=False, default=None
)
_args = vars(_parser.parse_args())

_SPECIFIED_BACKEND = _args["backend"]
_DEFAULT_BACKENDS = [
    "minidiff.backend.cupy",
    "minidiff.backend.mlx",
    "minidiff.backend.numpy",
]

current_backend = None


def import_backend(backend_name: str, package_name: Optional[str] = None):
    # https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
    module = importlib.import_module(backend_name, package=package_name)
    module_dict = module.__dict__
    return module_dict


def attempt_import(possible_backend: Optional[str]) -> Optional[dict]:
    if possible_backend is None:
        return None
    try:
        return import_backend(possible_backend)
    except:
        return None


def attempt_backend_import():
    global current_backend

    used_backend = None
    backend_exports = None

    backend_exports = attempt_import(_SPECIFIED_BACKEND)
    if backend_exports is not None:
        used_backend = _SPECIFIED_BACKEND
    else:
        for possible_backend in _DEFAULT_BACKENDS:
            backend_exports = attempt_import(possible_backend)
            if backend_exports is not None:
                used_backend = possible_backend
                break

    for export in backend_exports.values():
        if (
            isinstance(export, type)
            and export != type(Backend)
            and issubclass(export, Backend)
        ):
            if _SPECIFIED_BACKEND is not None and _SPECIFIED_BACKEND != used_backend:
                print(
                    f"could not find backend named {_SPECIFIED_BACKEND}, defaulting to {used_backend} instead"
                )
            current_backend = export
            return

    if current_backend is None or used_backend is None:
        raise Exception("could not find a suitable backend")


class Backend:
    pass
