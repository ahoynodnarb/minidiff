from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import graphviz

import minidiff as md

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

    import minidiff.typing as mdt


def draw_tensor_op_graph(
    root: md.Tensor,
    tensor_names: Optional[Dict[int, str]] = None,
    graph: Optional[graphviz.Graph] = None,
    insert_intermediates: bool = False,
    **kwargs,
) -> graphviz.Graph:
    # this essentially just finds the name of every input tensor
    # and lists them as arguments to the function which produced tensor
    def find_nested_tensor_name(tensor: md.Tensor) -> str:
        node = tensor.func_node
        input_names = []
        for input_tensor in node.op_inputs:
            input_names.append(lookup_tensor_name(input_tensor))
        op_name = node.op_name
        nested_tensor_name = f"{op_name}({', '.join(input_names)})"
        return nested_tensor_name

    def lookup_tensor_name(tensor: md.Tensor) -> str:
        nonlocal all_tensor_names
        nonlocal n_anonymous_tensors

        tensor_id = id(tensor)
        if isinstance(tensor, md.Tensor) and tensor.size == 1:
            tensor = tensor.item()
            tensor_id = id(tensor)
        # already has a name
        if tensor_id in all_tensor_names:
            tensor_name = all_tensor_names[tensor_id]
        # this is just a scalar so return the value as its name
        elif not isinstance(tensor, md.Tensor):
            tensor_name = str(tensor)
            all_tensor_names[tensor_id] = tensor_name
        # if we're either giving everything a name, or we haven't found its name and it's a leaf
        # then we give it a name
        elif insert_intermediates or tensor.is_leaf:
            tensor_name = f"t{n_anonymous_tensors}"
            n_anonymous_tensors += 1
            all_tensor_names[tensor_id] = tensor_name
        # default to just the explicit definition
        else:
            tensor_name = find_nested_tensor_name(tensor)
            all_tensor_names[tensor_id] = tensor_name

        return tensor_name

    # self-explanatory: just connect every tensor to the tensors which created it
    def add_edges(graph: graphviz.Graph, tensor: md.Tensor):
        if tensor.is_leaf:
            return
        node = tensor.func_node
        tensor_id = id(tensor)
        for child in node.op_inputs:
            child_id = id(child)
            graph.edge(str(child_id), str(tensor_id))

    def draw_tensor_graph(graph: graphviz.Graph, t: md.Tensor):
        # iterate through every tensor starting from leaf tensors
        all_tensors = t.toposort()
        for tensor in all_tensors:
            tensor_id = id(tensor)

            tensor_name = lookup_tensor_name(tensor)

            # if we're naming everything, then all tensors are expanded
            # otherwise, the tensor must be explicitly named to be expanded
            should_expand = insert_intermediates or tensor_id in tensor_names
            if not tensor.is_leaf and should_expand:
                tensor_name = f"{tensor_name} = {find_nested_tensor_name(tensor)}"

            graph.node(str(tensor_id), tensor_name)
            add_edges(graph, tensor)

    if graph is None:
        graph = graphviz.Digraph(**kwargs)

    if tensor_names is None:
        insert_intermediates = True
        tensor_names = {}

    n_anonymous_tensors = 0
    all_tensor_names = tensor_names.copy()

    draw_tensor_graph(graph, root)
    return graph


def calculate_finite_differences(
    *input_tensors: md.Tensor,
    func: mdt.GenericOp,
    h: float = 1e-7,
    exclude: Optional[List[md.Tensor]] = None,
) -> List[md.Tensor]:
    manual_gradients = []
    if exclude is None:
        exclude = []
    excluded_ids = [id(x) for x in exclude]
    with md.no_grad():
        for i, input_tensor in enumerate(input_tensors):
            if (
                not isinstance(input_tensor, md.Tensor)
                or not input_tensor.allow_grad
                or id(input_tensor) in excluded_ids
            ):
                manual_gradients.append(None)
                continue
            # this just computes the gradients from first principles
            left = input_tensors[:i]
            right = input_tensors[i + 1 :]
            flattened_input_tensor = input_tensor.reshape(-1)
            flattened_grad = md.zeros_like(flattened_input_tensor)
            # this is the same as (f(x + h) - f(x - h)) / (2 * h), which is the definition of the derivative
            for x in range(input_tensor.size):
                shifted_left = flattened_input_tensor.detach()
                shifted_left[x] += h
                shifted_left = shifted_left.reshape(input_tensor.shape)

                shifted_right = flattened_input_tensor.detach()
                shifted_right[x] -= h
                shifted_right = shifted_right.reshape(input_tensor.shape)

                first_term = func(*left, shifted_left, *right)
                second_term = func(*left, shifted_right, *right)
                calculated_grad = (first_term - second_term) / (2 * h)

                flattened_grad[x] = calculated_grad

            manual_gradients.append(flattened_grad.reshape(input_tensor.shape))

    return manual_gradients


# little helper function that just gives you the finite difference-calculated gradients and minidiff gradients
def compute_grads(
    *input_tensors: md.Tensor,
    func: mdt.GenericOp,
    h: float = 1e-7,
    exclude: Optional[List[md.Tensor]] = None,
) -> Tuple[List[md.Tensor], List[md.Tensor]]:
    if exclude is None:
        exclude = []

    excluded_ids = [id(x) for x in exclude]
    copied_input_tensors = []
    copied_exclude = []

    for t in input_tensors:
        copied = t.detach(allow_grad=True) if isinstance(t, md.Tensor) else deepcopy(t)
        copied_input_tensors.append(copied)
        if id(t) in excluded_ids:
            copied_exclude.append(copied)

    manual_gradients = calculate_finite_differences(
        *copied_input_tensors, func=func, h=h, exclude=copied_exclude
    )

    computed = func(*copied_input_tensors)
    computed.backward()
    automatic_gradients = [
        t.grad if isinstance(t, md.Tensor) else None for t in copied_input_tensors
    ]

    return manual_gradients, automatic_gradients
