import importlib


def import_selected_backend(selected_backend, package_name=None):
    # https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
    module = importlib.import_module(selected_backend, package=package_name)
    module_dict = module.__dict__
    if "__all__" in module_dict:
        module_exports = module_dict["__all__"]
    else:
        module_exports = [x for x in module_dict if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in module_exports})


found_backend = False

if not found_backend:
    try:
        import_selected_backend("minidiff.backend.mlx")
        found_backend = True
    except ImportError:
        pass

if not found_backend:
    try:
        import_selected_backend("minidiff.backend.numpy")
        found_backend = True
    except ImportError:
        pass

if not found_backend:
    raise Exception("could not find a suitable backend")

import importlib


def import_selected_backend(selected_backend, package_name=None):
    # https://stackoverflow.com/questions/43059267/how-to-do-from-module-import-using-importlib
    module = importlib.import_module(selected_backend, package=package_name)
    module_dict = module.__dict__
    if "__all__" in module_dict:
        module_exports = module_dict["__all__"]
    else:
        module_exports = [x for x in module_dict if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in module_exports})


found_backend = False

# if not found_backend:
#     try:
#         import_selected_backend("minidiff.backend.mlx")
#         found_backend = True
#     except ImportError:
#         pass

if not found_backend:
    try:
        import_selected_backend("minidiff.backend.numpy")
        found_backend = True
    except ImportError:
        pass

if not found_backend:
    raise Exception("could not find a suitable backend")
