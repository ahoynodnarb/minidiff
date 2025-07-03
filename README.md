# minidiff: A NumPy/CuPy-Compatible Reverse-Mode Automatic Differentiation Engine

A lightweight automatic differentiation library with PyTorch-like API, supporting higher-order gradients and eager memory management. 
Designed as a drop-in replacement for NumPy/CuPy with readability and clarity as a top priority.

This codebase does assume the reader has some existing requisite knowledge about multivariable calculus and linear algebra, but it's meant to be as readable as possible with that in mind. You can find great resources for both of those topics pretty much anywhere on the internet!

## Usage
Finding partial derivatives for almost any function is pretty simple, just:

Define some variables
```
x = md.Tensor([[0, 2, -2, 1], [-1, -1, -2, -2]], allow_grad=True)
y = md.Tensor([[2, 3, 4, 5], [0, -1, -3, 2]], allow_grad=True)
```
Run a calculation
```
f = 2 * y * md.sin(x) - x**2
```
And call `backward()`
```
f.backward(allow_higher_order=True)
# first order partial derivatives
# df/dx
print(x.grad)
# df/dy
print(y.grad)
```
If you want to go a little deeper and compute higher order partial derivatives, it's as simple as just doing a backward pass on the gradient tensors
```
x.grad.backward()
print("second order")
# now d^2f/dx^2
print(x.grad)
# now d^2f/dxdy
print(y.grad)
```
## Custom Functions
Implementing your own differentiable operations is also pretty simple. 

Helper functions like `create_op_func()` and `as_minidiff()` automatically wrap NumPy functions as minidiff functions and manage computation graphs, respectively. 

See [`minidiff/ops/definitions.py`](https://github.com/ahoynodnarb/minidiff/blob/master/minidiff/ops/definitions.py) for some examples.

## Project Structure
Feel free to explore around the repo, it's structured as follows:

[`minidiff/tensor.py`](https://github.com/ahoynodnarb/minidiff/blob/master/minidiff/tensor.py): Contains all the basic tensor code which includes a few convenient tensor creation functions and the actual backward pass

[`minidiff/topology.py`](https://github.com/ahoynodnarb/minidiff/blob/master/minidiff/topology.py): The `FuncNode` class is defined here; it mostly acts as a dataclass for graph nodes, but also handles gradient accumulation

[`minidiff/typing.py`](https://github.com/ahoynodnarb/minidiff/blob/master/minidiff/typing.py): Just extra types for convenience to keep type hinting clean and readable

[`minidiff/utils.py`](https://github.com/ahoynodnarb/minidiff/blob/master/minidiff/utils.py): A few utilities used throughout the codebase ranging from finite-difference testing to a graph visualization tool

[`minidiff/ops/definitions.py`](https://github.com/ahoynodnarb/minidiff/blob/master/minidiff/ops/definitions.py): Every operation exported by minidiff is defined here, including their forward and backward passes

[`minidiff/ops/wrapping.py`](https://github.com/ahoynodnarb/minidiff/blob/master/minidiff/ops/wrapping.py): This file mostly contains code that allows wraping NumPy functions as `Tensor` functions and automatic differentiability to those resulting `Tensor` functions