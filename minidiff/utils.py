import graphviz
import minidiff as md


def draw_tensor_op_graph(
    root, tensor_names=None, graph=None, insert_intermediates=False, **kwargs
):
    def find_nested_tensor_name(tensor):
        # this essentially just finds the name of every input tensor,
        # and lists them as arguments to the function which produced tensor
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
        # already has a name
        if tensor_id in all_tensor_names:
            tensor_name = all_tensor_names[tensor_id]
        # this is just a scalar
        elif tensor.size == 1:
            tensor_name = str(tensor.item())
            all_tensor_names[tensor_id] = tensor_name
        # if we're either giving everything a name, or we haven't found its name and it's a leaf
        elif insert_intermediates or tensor.is_leaf:
            tensor_name = f"t{n_anonymous_tensors}"
            n_anonymous_tensors += 1
            all_tensor_names[tensor_id] = tensor_name
        # default to just the explicit definition
        else:
            tensor_name = find_nested_tensor_name(tensor)
            all_tensor_names[tensor_id] = tensor_name

        return tensor_name

    def add_edges(graph, tensor):
        # self-explanatory: just connect every tensor to the tensors which created it
        if tensor.is_leaf:
            return
        node = tensor.func_node
        tensor_id = id(tensor)
        for child in node.input_tensors:
            child_id = id(child)
            graph.edge(str(child_id), str(tensor_id))

    def draw_tensor_graph(graph, t):
        # iterate through every tensor starting from leaf tensors
        all_tensors = t.toposort()
        for tensor in all_tensors:
            tensor_id = id(tensor)

            tensor_name = lookup_tensor_name(tensor)

            # if we're naming everything, then all tensors are expanded
            # otherwise, the tensor must be explicitly named to be expanded
            should_expand = tensor_id in tensor_names or insert_intermediates
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


def calculate_finite_differences(*input_tensors, func, h=1e-5):
    manual_gradients = []
    with md.no_grad():
        for i, input_tensor in enumerate(input_tensors):
            # this just computes the gradients from first principles
            left = input_tensors[:i]
            right = input_tensors[i + 1 :]
            flattened_input_tensor = input_tensor.reshape(-1)
            flattened_grad = md.zeros_like(flattened_input_tensor)
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


def compute_grads(*input_tensors, func):
    manual_gradients = calculate_finite_differences(*input_tensors, func=func)
    computed = func(*input_tensors)
    computed.backward()
    automatic_gradients = [t.grad for t in input_tensors]
    return manual_gradients, automatic_gradients
