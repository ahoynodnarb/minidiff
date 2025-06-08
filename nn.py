import math

try:
    import cupy as np  # type: ignore
except ImportError:
    import numpy as np

import minidiff as md
    
    
class Convolve2D:
    @staticmethod
    def add_padding(mat, padding=None):
        batch_size, height, width, channels = mat.shape

        if (
            padding is None
            or (isinstance(padding, tuple) and sum(padding) == 0)
            or padding == 0
        ):
            return mat

        if isinstance(padding, tuple):
            pad_top, pad_bottom, pad_left, pad_right = padding
        else:
            if padding % 1 == 0:
                pad_top = pad_bottom = pad_left = pad_right = padding
            else:
                padding = int(math.floor(padding))
                pad_top = pad_left = padding
                pad_bottom = pad_right = padding + 1
        pad_top, pad_bottom, pad_left, pad_right = int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)
        padded = np.zeros(
            (
                batch_size,
                pad_top + height + pad_bottom,
                pad_left + width + pad_right,
                channels,
            )
        )

        padded[:, pad_top : height + pad_top, pad_left : width + pad_left, :] = mat
        return padded

    @staticmethod
    def calculate_same_padding(height, width, kernel_height, kernel_width, stride):
        pad_vert = (height * (stride - 1) + kernel_height - stride) / 2
        pad_hori = (width * (stride - 1) + kernel_width - stride) / 2

        if pad_vert % 1 == 0:
            pad_top = pad_bottom = pad_vert
        else:
            pad_vert = int(math.floor(pad_vert))
            pad_top, pad_bottom = pad_vert, pad_vert + 1

        if pad_hori % 1 == 0:
            pad_left = pad_right = pad_hori
        else:
            pad_hori = int(math.floor(pad_hori))
            pad_left, pad_right = pad_hori, pad_hori + 1

        return (pad_top, pad_bottom, pad_left, pad_right)

    @staticmethod
    def calculate_convolved_dimensions(
        height, width, kernel_height, kernel_width, padding, stride
    ):
        if isinstance(padding, tuple):
            top, bottom, left, right = padding
            vertical_padding = top + bottom
            horizontal_padding = left + right
        else:
            vertical_padding = int(2 * padding)
            horizontal_padding = int(2 * padding)

        out_height = (height - kernel_height + vertical_padding) // stride + 1
        out_width = (width - kernel_width + horizontal_padding) // stride + 1
        return (int(out_height), int(out_width))

    @staticmethod
    def calculate_im2col_indices(rows_out, cols_out, kernel_height, kernel_width, stride):
        # these are the indices that correspond to each row within the patch
        kernel_row_indices = np.repeat(np.arange(kernel_height), kernel_width)
        # these are the indices corresponding to the row portion of the position of each patch within the input matrix
        conv_row_indices = stride * np.repeat(np.arange(rows_out), cols_out)

        # these are the indices that correspond to each column within the patch
        kernel_col_indices = np.tile(np.arange(kernel_width), kernel_height)
        # these are the indices that correspond to the column portion of the position of each patch within the input matrix
        conv_col_indices = stride * np.tile(np.arange(cols_out), rows_out)

        row_indices = kernel_row_indices.reshape((-1, 1)) + conv_row_indices.reshape(
            (1, -1)
        )
        col_indices = kernel_col_indices.reshape((-1, 1)) + conv_col_indices.reshape(
            (1, -1)
        )

        return (row_indices, col_indices)

    @staticmethod
    def forward(
        mat, kernels, padding=None, stride=1
    ):
        orig_shape = mat.shape
        batch_size, orig_height, orig_width, _ = orig_shape
        n_kernels, kernel_height, kernel_width, kernel_channels = kernels.shape

        out_dims = Convolve2D.calculate_convolved_dimensions(
            orig_height, orig_width, kernel_height, kernel_width, padding, stride
        )

        # out_dims is the "physical" dimension of the out matrix,
        # out_shape is the total shape which includes batch size and output channels
        out_shape = (batch_size, *out_dims, n_kernels)

        if padding is not None:
            mat = Convolve2D.add_padding(mat, padding=padding)

        # if we're not given the instructions on how to rearrange the matrix,
        # we can fall back to computing it manually
        # calculating the new positions for every element in the input image
        row_indices, col_indices = Convolve2D.calculate_im2col_indices(
            *out_dims, kernel_height, kernel_width, stride
        )

        # filter the input image by these new positions
        as_cols = mat[:, row_indices, col_indices, :]

        # flatten our matrix of kernels
        flattened_kernels = kernels.reshape(
            (n_kernels, kernel_height * kernel_width, kernel_channels)
        )

        # this is the actual convolution step, which is just a single matrix multiplication now!
        convolved = np.tensordot(as_cols, flattened_kernels, axes=((1, 3), (1, 2)))
        reshaped = convolved.reshape(out_shape)
        return reshaped

    @staticmethod
    def compute_grad_wrt_x(layer_input: md.Tensor, kernels: md.Tensor, grad: md.Tensor, padding=None, stride=1):
        # rotate kernels, then swap axes to match up correctly
        kernels_np = kernels._tensor
        grad_np = grad._tensor
        flipped_kernels = np.flip(np.flip(kernels_np, axis=1), axis=2)
        flipped_kernels = np.swapaxes(flipped_kernels, -1, 0)
        
        vertical_full_padding = (kernels_np.shape[1] - 1) // 2
        horizontal_full_padding = (kernels_np.shape[2] - 1) // 2
        full_padding = (
            vertical_full_padding,
            vertical_full_padding,
            horizontal_full_padding,
            horizontal_full_padding,
        )
        grad_wrt_x = Convolve2D.forward(
            grad_np,
            flipped_kernels,
            padding=full_padding,
            stride=1
        )
        return md.Tensor(grad_wrt_x, is_leaf=False)

    # https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
    # the gradient with respect to the weights (kernel) tells us how the loss function changes relative to
    # changes to each individual element of the kernel
    # the overall computation boils down to convolving each channel of the previous outputs by each channel of the gradient
    @staticmethod
    def compute_grad_wrt_w(layer_input: md.Tensor, kernels: md.Tensor, grad: md.Tensor, padding=None, stride=1):
        # normally, computing grad_wrt_w requires you to do convolutions for each slice of the previous outputs
        # and each slice of the gradient. But we can take advantage of batching to instead treat each slice of
        # output as a separate entry to the batch, and each slice of the gradient as a separate "kernel"
        # this results in us having the same final convolution, just the slices end up as the channels instead
        layer_input_np = layer_input._tensor
        grad_np = grad._tensor
        swapped_prev_outputs = np.swapaxes(layer_input_np, 0, -1)
        swapped_grad = np.swapaxes(grad_np, 0, -1)
        convolved = Convolve2D.forward(
            swapped_prev_outputs,
            swapped_grad,
            padding=padding,
            stride=stride
        )
        grad_wrt_w = np.swapaxes(convolved, 0, -1)

        return md.Tensor(grad_wrt_w, is_leaf=False)


class CrossEntropyLoss:
    @staticmethod
    def forward(y_true: md.Tensor, y_pred: md.Tensor, precompute_grad=False):
        if y_true is None:
            raise ValueError("Empty ground truth array")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        # avoid division by 0
        y_pred = y_pred.clip(a_min=1e-8, a_max=1 - 1e-8)
        # compute the one hot loss, reshape to match
        loss = -md.sum(y_true * md.log(y_pred), axis=-1, keepdims=True)
        return loss
    
    @staticmethod
    def loss(y_true: md.Tensor, y_pred: md.Tensor, grad, precompute_grad=False):
        y_pred = y_pred.clip(a_min=1e-8, a_max=1 - 1e-8)
        # more numerically stable than -y_true / y_pred
        if precompute_grad:
            return (y_pred - y_true) / len(y_true)
        return grad * -y_true / y_pred

convolve2d = md._generate_binary_op_func(forward_func=Convolve2D.forward, grad_a=Convolve2D.compute_grad_wrt_x, grad_b=Convolve2D.compute_grad_wrt_w, tensor_only=True, backend_op=True, propagate_kwargs=True)
cross_entropy_loss = md._generate_binary_op_func(forward_func=CrossEntropyLoss.forward, grad_a=None, grad_b=CrossEntropyLoss.loss, tensor_only=True, propagate_kwargs=True)

if __name__ == "__main__":
    import minidiff as md
    inputs = md.Tensor(
        [
            [
                [[1, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ],
            [
                [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ],
        ],
        allow_grad=True
    )
    # kernels need another dimension for channels
    kernels = md.Tensor(
        [
            [[[1, 0, 1], [1, 0, 1]], [[2, 0, 1], [2, 0, 1]]],
            [[[0, 1, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 1]]],
        ],
        allow_grad=True
    )
    y_pred = convolve2d(inputs, kernels, padding=1, stride=1)
    y_pred.backward()
    print(inputs.grad)
    print(kernels.grad)