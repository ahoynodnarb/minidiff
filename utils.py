try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np


from topology import FuncNode
import graphviz
import minidiff as md


def draw_tensor_op_graph(
    root, tensor_names=None, graph=None, insert_intermediates=False, **kwargs
):
    def find_nested_tensor_name(tensor):
        node = tensor.func_node
        input_names = []
        for input_tensor in node.input_tensors:
            input_names.append(lookup_tensor_name(input_tensor))
        op_name = node.op_name
        nested_tensor_name = f"{op_name}({', '.join(input_names)})"
        return nested_tensor_name

    def lookup_tensor_name(tensor):
        nonlocal all_tensor_names
        nonlocal n_anonymous_tensors

        tensor_id = id(tensor)
        if tensor_id in all_tensor_names:
            return all_tensor_names[tensor_id]
        elif tensor.size == 1:
            # this is just a scalar
            tensor_name = str(tensor.item())
            all_tensor_names[tensor_id] = tensor_name
        elif not insert_intermediates and not tensor.is_leaf and names_provided:
            tensor_name = find_nested_tensor_name(tensor)
            all_tensor_names[tensor_id] = tensor_name
        else:
            tensor_name = f"t{n_anonymous_tensors}"
            n_anonymous_tensors += 1
            all_tensor_names[tensor_id] = tensor_name

        return tensor_name

    def add_edges(graph, tensor):
        if not isinstance(tensor, md.Tensor):
            return
        if tensor.is_graph_source:
            return
        node = tensor.func_node
        tensor_id = id(tensor)
        for child in node.input_tensors:
            child_id = id(child)
            graph.edge(str(child_id), str(tensor_id))

    def draw_tensor_graph(graph, tensor):
        nonlocal names_provided

        tensor_id = id(tensor)
        if tensor_id in visited_tensors:
            return
        visited_tensors.add(tensor_id)

        tensor_name = lookup_tensor_name(tensor)

        # if names are not provided, all tensors are expanded
        # otherwise, the tensor must be explicitly named to be expanded
        is_named = not names_provided or tensor_id in tensor_names
        if not tensor.is_graph_source and (is_named or insert_intermediates):
            tensor_name = f"{tensor_name} = {find_nested_tensor_name(tensor)}"

        graph.node(str(tensor_id), tensor_name)

        if tensor.is_graph_source:
            return

        add_edges(graph, tensor)
        node = tensor.func_node
        for child in node.input_tensors:
            draw_tensor_graph(graph, child)

    visited_tensors = set()
    names_provided = tensor_names is not None
    n_anonymous_tensors = 0

    if graph is None:
        graph = graphviz.Digraph(**kwargs)

    if not names_provided:
        tensor_names = {}

    all_tensor_names = tensor_names.copy()

    draw_tensor_graph(graph, root)
    return graph


def calculate_finite_differences(*input_tensors, func, h=1e-8):
    manual_gradients = [0] * len(input_tensors)
    with md.no_grad():
        for i, input_tensor in enumerate(input_tensors):
            # this just computes the gradients from first principles
            left = input_tensors[:i]
            right = input_tensors[i + 1 :]
            first_term = func(*left, input_tensor + h, *right)
            second_term = func(*left, input_tensor, *right)

            calculated_grad = (first_term - second_term) / h
            manual_gradients[i] = calculated_grad

    return manual_gradients


def compute_grads(*input_tensors, func):
    manual_gradients = calculate_finite_differences(*input_tensors, func=func)
    computed = func(*input_tensors)
    computed.backward()
    automatic_gradients = [t.grad for t in input_tensors]
    return manual_gradients, automatic_gradients
