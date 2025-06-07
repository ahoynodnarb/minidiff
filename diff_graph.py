from tensor import Tensor
import minidiff

class FuncNode:
    def __init__(self, *inputs):
        self.inputs = inputs
        self.input_tensors = [x for x in inputs if isinstance(x, Tensor)]
        self.input_nodes = [x.diff_node for x in self.input_tensors]
        
    def update_grads(self, grad, wipe=False):
        raise NotImplementedError
    
    def pretty_repr(self):
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__} ({', '.join([str(x) for x in self.inputs])})"

class MatMulNode(FuncNode):
    def __init__(self, *inputs):
        assert len(inputs) == 2
        super().__init__(*inputs)
    
    def update_grads(self, grad, wipe=False):
        left = self.inputs[0]
        right = self.inputs[1]
        left.grad._tensor += grad.matmul(right.t)._tensor
        right.grad._tensor += left.t.matmul(grad)._tensor
        if wipe:
            self.inputs = None
    
    def pretty_repr(self):
        left = self.inputs[0]
        right = self.inputs[1]
        return f"{left} @ {right}"
    
class AddNode(FuncNode):
    def __init__(self, *inputs):
        assert len(inputs) == 2
        super().__init__(*inputs)
        
    def update_grads(self, grad, wipe=False):
        left = self.inputs[0]
        right = self.inputs[1]
        if isinstance(left, Tensor):
            left.grad._tensor += grad._tensor
        if isinstance(right, Tensor):
            right.grad._tensor += grad._tensor
        if wipe:
            self.inputs = None
    
    def pretty_repr(self):
        left = self.inputs[0]
        right = self.inputs[1]
        return f"{left} + {right}"
    
class SubtractNode(FuncNode):
    def __init__(self, *inputs):
        assert len(inputs) == 2
        super().__init__(*inputs)
        
    def update_grads(self, grad, wipe=False):
        left = self.inputs[0]
        right = self.inputs[1]
        if isinstance(left, Tensor):
            left.grad._tensor += grad._tensor
        if isinstance(right, Tensor):
            right.grad._tensor -= grad._tensor
        if wipe:
            self.inputs = None
    
    def pretty_repr(self):
        left = self.inputs[0]
        right = self.inputs[1]
        return f"{left} - {right}"
    
class MultiplyNode(FuncNode):
    def __init__(self, *inputs):
        assert len(inputs) == 2
        super().__init__(*inputs)
        
    def update_grads(self, grad, wipe=False):
        left = self.inputs[0]
        right = self.inputs[1]
        if isinstance(left, Tensor):
            left.grad._tensor += (grad * right)._tensor
        if isinstance(right, Tensor):
            right.grad._tensor += (grad * left)._tensor
        if wipe:
            self.inputs = None
    
    def pretty_repr(self):
        left = self.inputs[0]
        right = self.inputs[1]
        return f"{left} * {right}"
    
class TruedivNode(FuncNode):
    def __init__(self, *inputs):
        assert len(inputs) == 2
        super().__init__(*inputs)
        
    def update_grads(self, grad, wipe=False):
        left = self.inputs[0]
        right = self.inputs[1]
        if isinstance(left, Tensor):
            left.grad._tensor += (grad / right)._tensor
        if isinstance(right, Tensor):
            right.grad._tensor += (grad / left)._tensor
        if wipe:
            self.inputs = None
    
    def pretty_repr(self):
        left = self.inputs[0]
        right = self.inputs[1]
        return f"{left} / {right}"
    
class FloordivNode(FuncNode):
    def __init__(self, *inputs):
        assert len(inputs) == 2
        super().__init__(*inputs)
        
    def update_grads(self, grad, wipe=False):
        left = self.inputs[0]
        right = self.inputs[1]
        if isinstance(left, Tensor):
            left.grad._tensor += (grad // right)._tensor
        if isinstance(right, Tensor):
            right.grad._tensor += (grad // left)._tensor
        if wipe:
            self.inputs = None
    
    def pretty_repr(self):
        left = self.inputs[0]
        right = self.inputs[1]
        return f"{left} // {right}"
    
class PowNode(FuncNode):
    def __init__(self, *inputs):
        assert len(inputs) == 2
        super().__init__(*inputs)
        
    def update_grads(self, grad, wipe=False):
        t = self.inputs[0]
        n = self.inputs[1]
        t.grad._tensor += (grad * (n * t ** (n-1)))._tensor
        if wipe:
            self.inputs = None