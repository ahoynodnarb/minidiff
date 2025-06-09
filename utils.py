from topology import FuncNode
import graphviz
import minidiff as md


def draw_tensor_op_graph(root, tensor_names=None, graph=None, insert_intermediates=False):
    def find_nested_tensor_name(tensor):
        node = tensor.func_node
        input_names = []
        for input_tensor in node.input_tensors:
            input_names.append(
                lookup_tensor_name(input_tensor)
            )
        op_name = node.op_name
        nested_tensor_name = f"{op_name}({', '.join(input_names)})"
        return nested_tensor_name

    def lookup_tensor_name(tensor):
        nonlocal all_tensor_names
        nonlocal n_anonymous_tensors
        
        tensor_id = id(tensor)
        if tensor_id in all_tensor_names:
            return all_tensor_names[tensor_id]
        elif not insert_intermediates and not tensor.is_leaf and names_provided:
            tensor_name = find_nested_tensor_name(tensor)
            all_tensor_names[tensor_id] = tensor_name
        elif tensor.size == 1:
            # this is just a scalar
            tensor_name = str(tensor.item())
            all_tensor_names[tensor_id] = tensor_name
        else:
            tensor_name = f"t{n_anonymous_tensors}"
            n_anonymous_tensors += 1
            all_tensor_names[tensor_id] = tensor_name

        return tensor_name

    def add_edges(graph, tensor):
        if not isinstance(tensor, md.Tensor):
            return
        if tensor.is_leaf:
            return
        node = tensor.func_node
        # op_name = node.op_name
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
        if not tensor.is_leaf and (is_named or insert_intermediates):
            tensor_name = f"{tensor_name} = {find_nested_tensor_name(tensor)}"
            
        graph.node(str(tensor_id), tensor_name)

        if tensor.is_leaf:
            return

        add_edges(graph, tensor)
        node = tensor.func_node
        for child in node.input_tensors:
            draw_tensor_graph(graph, child)
            
    visited_tensors = set()
    names_provided = tensor_names is not None
    n_anonymous_tensors = 0

    if graph is None:
        graph = graphviz.Digraph()
        
    if not names_provided:
        tensor_names = {}
        
    all_tensor_names = tensor_names.copy()
        
    draw_tensor_graph(graph, root)
    return graph


if __name__ == "__main__":
    a = md.Tensor([[0, 2, -2, 1], [-1, -1, -2, -2]], allow_grad=True)
    b = md.Tensor([[1, 2, 3, 4], [4, 3, 2, 1]], allow_grad=True)
    c = md.Tensor([[0, 1, -1, 2], [1, 0, 1, 0]], allow_grad=True)
    d = b + c - a**2
    e = md.sin(d * c)
    f = 2 * e - b * d

    # f = 2 * e - b * d
    # f = 2 * sin(d * c) - b * (b + c - a**2)
    # f = 2 * sin((b + c - a**2) * c) - b**2 - b * c + b * a**2
    # f = 2 * sin(b * c + c**2 - c * a**2) - b**2 - b * c + b * a**2

if __name__ == "__main__":
    tensor_names = {
        id(a): "a",
        id(b): "b",
        id(c): "c",
        id(d): "d",
        id(e): "e",
        id(f): "f",
    }
    graph = draw_tensor_op_graph(f, tensor_names=tensor_names)
    graph.render("graph", view=True)
