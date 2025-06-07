# TODO: Topologically sort before traversal

try:
    import cupy as np # type: ignore
except ImportError:
    import numpy as np

import minidiff    
import diff_graph

# compute from left to right, dy/dw2 then dw2/dw1 to get dy/dw1 and finally dw1/dx to get dy/dx
# dy/dw2 would just be the loss gradient
    
class Tensor:
    def __init__(self, tensor, allow_grad=True):
        if isinstance(tensor, np.ndarray):
            self._tensor = tensor
        else:
            self._tensor = np.array(tensor)
        
        if allow_grad:
            self.grad = zeros_like(self, allow_grad=False)
            
        self.diff_node = None
        self._allow_grad = allow_grad
        
    @property
    def t(self):
        return self._tensor.t
    
    @property
    def shape(self):
        return self._tensor.shape
    
    @property
    def allow_grad(self):
        return self._allow_grad and minidiff.grad_allowed()
    
        
    def backward(self):
        if not self.allow_grad:
            return
        
        if self.diff_node is None:
            return
        
        seen = []
        stack = []
        # topologically sort
        def dfs(tensor):
            root = tensor.diff_node
            if root is None or root in seen:
                return
            seen.append(root)
            for input_node, input_tensor in zip(root.input_nodes, root.input_tensors):
                dfs(input_tensor)
            stack.append(tensor)
        self.grad = ones_like(self, allow_grad=self.allow_grad)
        dfs(self)
        for tensor in reversed(stack):
            n = tensor.diff_node
            n_grad = tensor.grad
            n.update_grads(n_grad)
            
        
        # def traverse_nodes(root: diff_graph.FuncNode, grad):
        #     if root is None:
        #         return
        #     # node in input_nodes will be None if it is just a variable, not a function, that means the tensors are variables
        #     root.update_grads(grad)
        #     for node, in_tensor in zip(root.input_nodes, root.input_tensors):
        #         traverse_nodes(node, in_tensor.grad)
                
        # self.grad = ones_like(self, allow_grad=False)
        # traverse_nodes(self.diff_node, self.grad)
    
    def reshape(self, *args, **kwargs):
        self._tensor = self._tensor.reshape(*args, **kwargs)
    
    def matmul(self, t2):
        return self @ t2
    
    def add(self, t2):
        return self + t2
    
    def multiply(self, t2):
        return self * t2
    
    def __matmul__(self, t2):
        return matmul(self, t2)
    
    def __add__(self, other):
        return add(self, other)
    
    def __sub__(self, other):
        return subtract(self, other)
    
    def __mul__(self, other):
        return multiply(self, other)
    
    def __rmul__(self, other):
        return multiply(self, other)
    
    def __truediv__(self, other):
        return truediv(self, other)
    
    def __floordiv__(self, other):
        return floordiv(self, other)
    
    def __repr__(self):
        return self._tensor.__repr__()
    
def ones_like(t1: Tensor, allow_grad=True):
    return Tensor(np.ones_like(t1._tensor), allow_grad=allow_grad)

def zeros_like(t1: Tensor, allow_grad=True):
    return Tensor(np.zeros_like(t1._tensor), allow_grad=allow_grad)
    
def matmul(t1: Tensor, other: Tensor, allow_grad=True):
    output = Tensor(np.matmul(t1._tensor, other._tensor), allow_grad=allow_grad)
    if allow_grad:
        output.diff_node = diff_graph.MatMulNode(t1, other)
    return output

def add(t1: Tensor, other, allow_grad=True):
    if isinstance(other, Tensor):
        output = Tensor(t1._tensor + other._tensor, allow_grad=allow_grad)
    else:
        output = Tensor(t1._tensor + other, allow_grad=allow_grad)
        
    if allow_grad:
        output.diff_node = diff_graph.AddNode(t1, other)
        
    return output

def subtract(t1: Tensor, other, allow_grad=True):
    if isinstance(other, Tensor):
        output = Tensor(t1._tensor - other._tensor, allow_grad=allow_grad)
    else:
        output = Tensor(t1._tensor - other, allow_grad=allow_grad)
        
    if allow_grad:
        output.diff_node = diff_graph.SubtractNode(t1, other)
        
    return output
    
def multiply(t1: Tensor, other, allow_grad=True):
    if isinstance(other, Tensor):
        output = Tensor(t1._tensor * other._tensor, allow_grad=allow_grad)
    else:
        output = Tensor(t1._tensor * other, allow_grad=allow_grad)
        
    if allow_grad:
        output.diff_node = diff_graph.MultiplyNode(t1, other)
        
    return output

def truediv(t1: Tensor, other, allow_grad=True):
    if isinstance(other, Tensor):
        output = Tensor(t1._tensor / other._tensor, allow_grad=allow_grad)
    else:
        output = Tensor(t1._tensor / other, allow_grad=allow_grad)
        
    if allow_grad:
        output.diff_node = diff_graph.TruedivNode(t1, other)
        
    return output

def floordiv(t1: Tensor, other, allow_grad=True):
    if isinstance(other, Tensor):
        output = Tensor(t1._tensor // other._tensor, allow_grad=allow_grad)
    else:
        output = Tensor(t1._tensor // other, allow_grad=allow_grad)
        
    if allow_grad:
        output.diff_node = diff_graph.TruedivNode(t1, other)
        
    return output

def power(t1: Tensor, n, allow_grad=True):
    output = Tensor(np.power(t1._tensor, n), allow_grad=allow_grad)
    if allow_grad:
        output.diff_node = diff_graph.PowNode(t1, n)
    return output

def floor(t1: Tensor, allow_grad=True):
    return Tensor(np.floor(t1._tensor), allow_grad=allow_grad)