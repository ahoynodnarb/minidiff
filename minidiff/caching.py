from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

import minidiff as md

if TYPE_CHECKING:
    from typing import Tuple

_caching_graph = ContextVar("caching_graph", default=False)
_cached_graph_indices = ContextVar("cached_indices", default=None)


class reuse_graph:
    def __enter__(self):
        self.prev_caching = _caching_graph.get()
        _caching_graph.set(True)
        _cached_graph_indices.set({})

    def __exit__(self, type, value, traceback):
        _caching_graph.set(self.prev_caching)
        _cached_graph_indices.set({})


def currently_caching() -> bool:
    return _caching_graph.get()


def indices_for_tensor(tensor: md.Tensor) -> Tuple[int, ...]:
    if not _caching_graph.get():
        raise ValueError("Not currently preserving graph")

    if tensor.is_leaf:
        return ()

    sorted_tensors = tensor.toposort()

    if not sorted_tensors:
        return ()

    tensor_hash = tensor.op_node.hash

    indices_dict = _cached_graph_indices.get()
    if tensor_hash in indices_dict:
        return indices_dict[tensor_hash]

    full_graph = tensor.op_node._tensor_graph
    tensor_to_index = {id(t): -1 for t in sorted_tensors if t is not tensor}

    stack = [([i], t) for i, t in enumerate(full_graph)]

    while stack:
        index_list, item = stack.pop()
        if isinstance(item, list):
            stack.extend((index_list + [i], t) for i, t in enumerate(item))
            continue

        if id(item) not in tensor_to_index:
            continue
        tensor_to_index[id(item)] = tuple(index_list)

    indices = tuple(tensor_to_index[id(t)] for t in sorted_tensors if t is not tensor)

    indices_dict[tensor_hash] = indices

    return indices
