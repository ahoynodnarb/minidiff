This is a very small automatic differentiation engine I'm trying to implement. For now, it just has plain NumPy and CuPy backends. So far it only supports basic operations (+,-,*,/); some basic functions (sin, cos, exp, pow); and matrix multiplication (untested).
If you want to implement your own operation, you can call `generate_binary_op_func` or `generate_unary_op_func` for functions of 2 or 1 variables, respectively, specify your partials for each of those variables, and you're done!
Everything should be handled by the Tensor class' built-in topological sorting and reverse-mode autodiff.
Just call `backward()` on your output tensor and it will update the `.grad` of each tensor in its computational graph.

Bear in mind this project is meant to be more-or-less a NumPy wrapper that implements clean automatic differentiation.