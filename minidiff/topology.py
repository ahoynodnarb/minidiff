from __future__ import annotations

from typing import TYPE_CHECKING

import minidiff as md
import minidiff.caching as mdc

if TYPE_CHECKING:
    from typing import Any, Dict, List, Literal, Optional, Sequence

    import minidiff.typing as mdt


# OpNodes represent operations on the computation graph, with incoming edges being the input tensors, and outgoing the output tensors
class OpNode:
    def __init__(
        self,
        forward_func: mdt.GenericOp,
        grad_functions: Sequence[Optional[mdt.GenericOpGrad]],
        op_inputs: Sequence[Any],
        op_kwargs: Optional[Dict[str, Any]] = None,
        op_name: Optional[str] = None,
        propagate_kwargs: bool = False,
    ):
        self.grad_functions = grad_functions

        self.op_inputs = op_inputs

        if op_kwargs is None:
            op_kwargs = {}
        self.op_kwargs = op_kwargs

        if op_name is None:
            op_name = ""
        self.op_name = op_name

        self.propagate_kwargs = propagate_kwargs

        self.tensor_inputs = [x for x in op_inputs if isinstance(x, md.Tensor)]

        for tensor in self.tensor_inputs:
            tensor.graph_refs += 1

        self._tensor_graph = []

        if not mdc.currently_caching():
            self._op_ids = []
            return

        self._op_ids = [None for _ in range(len(self.op_inputs) + 1)]

        # since order of the inputs matters, substitute -1 for non-tensor/leaf op_inputs
        for i, op_input in enumerate(self.op_inputs):
            if not isinstance(op_input, md.Tensor) or op_input.is_leaf:
                self._op_ids[i] = -1
            else:
                self._op_ids[i] = op_input.op_node._op_ids

        self._op_ids[-1] = id(forward_func)
        self._op_ids = tuple(self._op_ids)

        seen = set()

        for op_input in self.op_inputs:
            if not isinstance(op_input, md.Tensor):
                continue
            if id(op_input) in seen:
                continue
            if not op_input.is_leaf:
                self._tensor_graph.append(op_input.op_node._tensor_graph)

            self._tensor_graph.append(op_input)

            seen.add(id(op_input))

    @property
    def hash(self) -> int:
        return hash(self._op_ids)

    # this accumulates gradients for the input tensors through chain rule (reverse-mode)
    def update_grads(self, grad: md.Tensor):
        # don't use no_grad() here because we are assuming gradients already don't track their gradients,
        # and if they do, they may be doing higher-order partial derivatives
        for op_input, grad_function in zip(self.op_inputs, self.grad_functions):
            if not isinstance(op_input, md.Tensor):
                continue
            if not op_input.allow_grad:
                continue
            if grad_function is None:
                continue

            kwargs = self.op_kwargs if self.propagate_kwargs else {}
            grad_computation = grad_function(*self.op_inputs, grad, **kwargs)
            # if broadcasting occured during the forward pass, we need to collect gradients
            # back in the backward pass so that the gradients are correctly distributed
            if grad_computation.shape == op_input.shape:
                collected_grad = grad_computation
            else:
                collected_grad = md.unbroadcast(grad_computation, op_input.shape)

            if op_input.grad is None:
                op_input.grad = collected_grad
            else:
                op_input.grad = op_input.grad + collected_grad

    def toposort(self) -> List[md.Tensor]:
        seen = set()
        traversal_path = []

        # topologically sort:
        # step through the graph starting from the output tensor (self)
        # go all the way down to the leaf tensors, skipping tensors we've already seen
        # after getting all the way to the base, finally push ourselves onto the stack
        # rinse and repeat for the input tensors, their input tensors, etc.
        def dfs(op_node: OpNode):
            if op_node is None:
                return
            for op_input in op_node.tensor_inputs:
                input_id = id(op_input)
                if input_id in seen:
                    continue
                seen.add(input_id)
                dfs(op_input.op_node)
                traversal_path.append(op_input)

        dfs(self)

        return traversal_path

    # this does the actual advertised reverse-mode automatic differentiation.
    # I mostly just referenced this Wikipedia page: https://en.wikipedia.org/wiki/Automatic_differentiation
    def backward(
        self,
        seed_grad: md.Tensor,
        retain_grads: bool = False,
        cleanup_mode: Literal["keep", "prune", "destroy"] = "prune",
        allow_higher_order: bool = False,
        reset_grads: bool = True,
    ):
        if cleanup_mode not in ["keep", "prune", "destroy"]:
            cleanup_mode = "prune"

        # computing higher order derivatives means partially re-traversing the subgraph for whichever variable
        # we're computing the higher order derivative of, so the graph needs to remain.
        # in accumulating gradients when calling backward() the second time, gradients from intermediates
        # will almost always be necessary so those have to be kept in memory too
        if allow_higher_order:
            retain_grads = True
            if cleanup_mode == "destroy":
                cleanup_mode = "prune"

        if mdc.currently_caching():
            full_graph = self._tensor_graph

            traversal_indices = mdc.backward_indices_for_root(self)
            traversal_path = [None] * len(traversal_indices)

            for i, indices in enumerate(traversal_indices):
                current_item = full_graph
                for index in indices:
                    current_item = current_item[index]
                traversal_path[i] = current_item
        else:
            traversal_path = self.toposort()

        if reset_grads:
            for tensor in traversal_path:
                tensor.grad = None

        with md.enable_grad(allow_higher_order):
            self.update_grads(seed_grad)
            for tensor in reversed(traversal_path):
                # leaf tensors don't have any input tensors to update, so skip
                if tensor.is_leaf:
                    continue
                # this should never be None since the final gradient (self's gradient) is manually set to ones
                # first iteration updates input tensors who now have non-None grads too
                # this continues for their input tensors, and those tensor's inputs, and so on and so forth
                grad = tensor.grad
                node = tensor.op_node
                node.update_grads(grad)
                # we're only temporarily storing grads
                # so we need to remove any references when we're done to save memory
                if not retain_grads:
                    tensor.grad = None

                if cleanup_mode == "keep":
                    continue

                if cleanup_mode == "destroy":
                    tensor.wipe()
                    continue

                if tensor.graph_refs > 0:
                    continue

                for child in node.tensor_inputs:
                    child.graph_refs -= 1

                tensor.wipe()

    def __repr__(self) -> str:
        return f"{self.op_name}({', '.join([str(x) for x in self.op_inputs])})"
