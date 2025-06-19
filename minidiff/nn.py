from __future__ import annotations

import collections
import math
from typing import Optional, Sequence, Tuple, Union

import minidiff as md
import minidiff.ops as ops
import minidiff.typing as mdt
from minidiff.utils import get_exported_var_names


class Convolve2D(ops.BinaryOpClass):
    @staticmethod
    def get_padded_edges(padding):
        # padding is already a tuple
        if isinstance(padding, collections.Sequence):
            return padding
        # padding is an integer
        if padding % 1 == 0:
            return (padding, padding, padding, padding)
        # padding is a float
        padding = int(math.floor(padding))
        pad_top = pad_left = padding
        pad_bottom = pad_right = padding + 1

        return pad_top, pad_bottom, pad_left, pad_right

    @staticmethod
    @ops.unary_op_func(
        grad=lambda a, grad, padding=None: Convolve2D.add_padding(
            grad, padding=padding
        ),
        propagate_kwargs=True,
        casting=None,
    )
    def remove_padding(
        mat: md.Tensor, padding: Optional[Union[int, float, Sequence[int]]] = None
    ) -> md.Tensor:
        _, height, width, _ = mat.shape

        if (
            padding is None
            or (isinstance(padding, collections.Sequence) and sum(padding) == 0)
            or padding == 0
        ):
            return mat

        pad_top, pad_bottom, pad_left, pad_right = Convolve2D.get_padded_edges(padding)

        # return view of padded matrix cropped around the padded boundaries
        return mat[:, pad_top : height - pad_bottom, pad_left : width - pad_right, :]

    @staticmethod
    @ops.unary_op_func(
        grad=lambda a, grad, padding=None: Convolve2D.remove_padding(
            grad, padding=padding
        ),
        propagate_kwargs=True,
        casting=None,
    )
    def add_padding(
        mat: md.Tensor, padding: Optional[Union[int, float, Sequence[int]]] = None
    ) -> md.Tensor:
        batch_size, height, width, channels = mat.shape

        if (
            padding is None
            or (isinstance(padding, collections.Sequence) and sum(padding) == 0)
            or padding == 0
        ):
            return mat

        pad_top, pad_bottom, pad_left, pad_right = Convolve2D.get_padded_edges(padding)

        padded = md.zeros(
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
    def calculate_same_padding(
        height: int, width: int, kernel_height: int, kernel_width: int, stride: int
    ) -> Tuple[int, int, int, int]:
        pad_vert = (height * (stride - 1) + kernel_height - stride) / 2
        pad_hori = (width * (stride - 1) + kernel_width - stride) / 2

        # we can evenly pad vertically
        if pad_vert % 1 == 0:
            pad_top = pad_bottom = pad_vert
        else:
            pad_vert = int(math.floor(pad_vert))
            pad_top, pad_bottom = pad_vert, pad_vert + 1

        # we can evenly pad horizontally
        if pad_hori % 1 == 0:
            pad_left = pad_right = pad_hori
        else:
            pad_hori = int(math.floor(pad_hori))
            pad_left, pad_right = pad_hori, pad_hori + 1

        return (pad_top, pad_bottom, pad_left, pad_right)

    # formula for full padding is just kernel_dim - original_pad_dim - 1
    @staticmethod
    def calculate_full_padding(
        kernel_height: int,
        kernel_width: int,
        original_padding: Union[int, float, Sequence[int]],
    ) -> Tuple[int, int, int, int]:
        pad_top = kernel_height - 1
        pad_bottom = kernel_height - 1
        pad_left = kernel_width - 1
        pad_right = kernel_width - 1
        if isinstance(original_padding, collections.Sequence):
            o_top, o_bottom, o_left, o_right = original_padding
            pad_top -= o_top
            pad_bottom -= o_bottom
            pad_left -= o_left
            pad_right -= o_right
        else:
            pad_top -= original_padding
            pad_bottom -= original_padding
            pad_left -= original_padding
            pad_right -= original_padding
        return (pad_top, pad_bottom, pad_left, pad_right)

    # formula for a convolved dimension is just (dim - kernel_dim + 2 * padding) / stride + 1
    @classmethod
    def calculate_convolved_dimensions(
        cls,
        height: int,
        width: int,
        kernel_height: int,
        kernel_width: int,
        padding: Union[int, float, Sequence[int]],
        stride: int,
    ) -> Tuple[int, int]:
        if isinstance(padding, collections.Sequence):
            top, bottom, left, right = padding
            vertical_padding = top + bottom
            horizontal_padding = left + right
        else:
            vertical_padding = int(2 * padding)
            horizontal_padding = int(2 * padding)

        out_height = (height - kernel_height + vertical_padding) // stride + 1
        out_width = (width - kernel_width + horizontal_padding) // stride + 1
        return (int(out_height), int(out_width))

    # this returns indices that we use to index over a matrix of rows_out x cols_out so that it is im2col'ed
    @staticmethod
    def calculate_im2col_indices(
        rows_out: int, cols_out: int, kernel_height: int, kernel_width: int, stride: int
    ) -> Tuple[md.Tensor, md.Tensor]:
        # these are the indices that correspond to each row within the patch
        kernel_row_indices = md.repeat(md.arange(kernel_height), kernel_width)
        # these are the indices corresponding to the row portion of the position of each patch within the input matrix
        conv_row_indices = stride * md.repeat(md.arange(rows_out), cols_out)

        # these are the indices that correspond to each column within the patch
        kernel_col_indices = md.tile(md.arange(kernel_width), kernel_height)
        # these are the indices that correspond to the column portion of the position of each patch within the input matrix
        conv_col_indices = stride * md.tile(md.arange(cols_out), rows_out)

        row_indices = kernel_row_indices.reshape((-1, 1)) + conv_row_indices.reshape(
            (1, -1)
        )
        col_indices = kernel_col_indices.reshape((-1, 1)) + conv_col_indices.reshape(
            (1, -1)
        )

        return (row_indices, col_indices)

    # this transforms the input tensor into a tensor where the columns make up each window of the convolution
    # that way we can just perform a tensordot with the partially flattened kernels to simulate a convolution much faster
    @classmethod
    def perform_convolution(
        cls,
        mat: md.Tensor,
        kernels: md.Tensor,
        padding: Optional[Union[int, float, Sequence[int]]] = None,
        stride: int = 1,
        im2col_indices: Optional[Tuple[md.Tensor, md.Tensor]] = None,
        out_dims: Optional[Sequence[int]] = None,
    ) -> md.Tensor:
        orig_shape = mat.shape
        batch_size, orig_height, orig_width, _ = orig_shape
        n_kernels, kernel_height, kernel_width, kernel_channels = kernels.shape
        if out_dims is None:
            out_dims = cls.calculate_convolved_dimensions(
                orig_height, orig_width, kernel_height, kernel_width, padding, stride
            )

        # out_dims is the "physical" dimension of the out matrix,
        # out_shape is the total shape which includes batch size and output channels
        out_shape = (batch_size, *out_dims, n_kernels)

        if padding is not None:
            mat = cls.add_padding(mat, padding=padding)

        # if we're not given the instructions on how to rearrange the matrix,
        # we can fall back to computing it manually
        if im2col_indices is None:
            # calculating the new positions for every element in the input image
            row_indices, col_indices = cls.calculate_im2col_indices(
                *out_dims, kernel_height, kernel_width, stride
            )
        else:
            row_indices, col_indices = im2col_indices

        # filter the input image by these new positions
        as_cols = mat[:, row_indices, col_indices, :]

        # flatten our matrix of kernels
        flattened_kernels = kernels.reshape(
            (n_kernels, kernel_height * kernel_width, kernel_channels)
        )

        # this is the actual convolution step, which is just a single matrix multiplication now!
        convolved = md.tensordot(as_cols, flattened_kernels, axes=((1, 3), (1, 2)))
        reshaped = convolved.reshape(out_shape)
        return reshaped

    def setup(
        self,
        conv_input: md.Tensor,
        kernels: md.Tensor,
        padding: Union[int, float, Sequence[int]] = 0,
        stride: int = 1,
    ):
        _, in_height, in_width, self.in_channels = conv_input.shape
        self.n_kernels, self.kernel_height, self.kernel_width, _ = kernels.shape

        if isinstance(padding, collections.Sequence):
            pad_top, pad_bottom, pad_left, pad_right = padding
        else:
            pad_top = pad_bottom = pad_left = pad_right = padding
        if (in_height - self.kernel_height + pad_top + pad_bottom) % stride != 0:
            raise ValueError("Cannot evenly convolve")
        if (in_height - self.kernel_width + pad_left + pad_right) % stride != 0:
            raise ValueError("Cannot evenly convolve")

        # we need to keep track of the shape of the inputs and outputs so we do not
        # have to recalculate them for every single batch
        self.in_dims = (in_height, in_width)
        self.out_dims = self.calculate_convolved_dimensions(
            in_height, in_width, self.kernel_height, self.kernel_width, padding, stride
        )

        self.padding = padding
        self.stride = stride

        # we optimize the actual convolution as a large matrix multiplication
        # and we keep track of how the matrices need to be rearranged for that
        # matrix multiplication, also so we don't have to recompute it for each batch
        self.forward_indices = self.calculate_im2col_indices(
            *self.out_dims, self.kernel_height, self.kernel_width, self.stride
        )
        self.backward_input_indices = self.calculate_im2col_indices(
            *self.in_dims, self.kernel_height, self.kernel_width, self.stride
        )
        self.backward_kern_indices = self.calculate_im2col_indices(
            self.kernel_height, self.kernel_width, *self.out_dims, self.stride
        )

    def create_forward(self) -> mdt.BinaryFunc:
        def forward(
            conv_input: md.Tensor,
            kernels: md.Tensor,
            padding: Union[int, float, Sequence[int]] = 0,
            stride: int = 1,
        ) -> md.Tensor:
            self.setup(
                conv_input=conv_input,
                kernels=kernels,
                padding=padding,
                stride=stride,
            )
            convolved = self.perform_convolution(
                conv_input,
                kernels,
                padding=self.padding,
                stride=self.stride,
                out_dims=self.out_dims,
                im2col_indices=self.forward_indices,
            )
            return convolved

        return forward

    def create_grads(self) -> Tuple[mdt.BinaryOpGrad, mdt.BinaryOpGrad]:
        def compute_grad_wrt_x(
            conv_input: md.Tensor,
            kernels: md.Tensor,
            grad: md.Tensor,
        ) -> md.Tensor:
            # rotate kernels, then swap axes to match up correctly
            flipped_kernels = md.flip(md.flip(kernels, axis=1), axis=2)
            flipped_kernels = md.swapaxes(flipped_kernels, -1, 0)

            full_padding = self.calculate_full_padding(
                kernel_height=self.kernel_width,
                kernel_width=self.kernel_height,
                original_padding=self.padding,
            )
            grad_wrt_x = Convolve2D.perform_convolution(
                grad,
                flipped_kernels,
                padding=full_padding,
                stride=1,
                out_dims=self.in_dims,
                im2col_indices=self.backward_input_indices,
            )

            return grad_wrt_x

        # https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
        # the gradient with respect to the weights (kernel) tells us how the loss function changes relative to
        # changes to each individual element of the kernel
        # the overall computation boils down to convolving each channel of the previous outputs by each channel of the gradient
        def compute_grad_wrt_w(
            conv_input: md.Tensor,
            kernels: md.Tensor,
            grad: md.Tensor,
        ) -> md.Tensor:
            # normally, computing grad_wrt_w requires you to do convolutions for each slice of the previous outputs
            # and each slice of the gradient. But we can take advantage of batching to instead treat each slice of
            # output as a separate entry to the batch, and each slice of the gradient as a separate "kernel"
            # this results in us having the same final convolution, just the slices end up as the channels instead
            swapped_prev_outputs = md.swapaxes(conv_input, 0, -1)
            swapped_grad = md.swapaxes(grad, 0, -1)
            convolved = Convolve2D.perform_convolution(
                swapped_prev_outputs,
                swapped_grad,
                padding=self.padding,
                stride=self.stride,
                out_dims=(self.kernel_height, self.kernel_width),
                im2col_indices=self.backward_kern_indices,
            )
            grad_wrt_w = md.swapaxes(convolved, 0, -1)

            return grad_wrt_w

        return (compute_grad_wrt_x, compute_grad_wrt_w)


class CrossEntropyLoss(ops.BinaryOpClass):
    def create_forward(self) -> mdt.BinaryFunc:
        # formula for cross entropy loss is sum(y_true * -log(y_pred))
        def loss_func(
            y_true: md.Tensor,
            y_pred: md.Tensor,
            precompute_grad: bool = False,
            smoothing: Union[int, float] = 0,
        ) -> md.Tensor:
            if y_true is None:
                raise ValueError("Empty ground truth array")
            if y_true.shape != y_pred.shape:
                raise ValueError("y_true and y_pred must have the same shape")
            n_classes = y_true.shape[-1]
            y_smoothed = (1 - smoothing) * y_true + (smoothing / n_classes)
            # avoid division by 0
            y_pred = y_pred.clip(a_min=1e-8, a_max=None)
            # compute the one hot loss, reshape to match
            loss = md.sum(y_smoothed * -md.log(y_pred), axis=-1, keepdims=True)
            return loss

        return loss_func

    def create_grads(self) -> Tuple[None, mdt.BinaryOpGrad]:
        # partial derivative of cross entropy loss wrt to predictions is pretty easy to derive: -y_true / y_pred
        # if we want to precompute the gradient (using logsoftmax) the gradient calculation becomes (y_pred - y_true)
        # we just throw in the division by len(y_true) to make up for batching
        # precomputing the gradient (using logsoftmax) is just a lot more numerically stable since there's no risk of division by 0
        def loss_gradient(
            y_true: md.Tensor,
            y_pred: md.Tensor,
            grad,
            precompute_grad: bool = False,
            smoothing: Union[int, float] = 0,
        ) -> md.Tensor:
            n_classes = y_true.shape[-1]
            y_smoothed = (1 - smoothing) * y_true + (smoothing / n_classes)
            y_pred = y_pred.clip(a_min=1e-8, a_max=None)
            # more numerically stable than -y_true / y_pred
            if precompute_grad:
                return grad * (y_pred - y_smoothed) / len(y_smoothed)
            return grad * -y_smoothed / y_pred

        return (None, loss_gradient)


exported_ops = [
    convolve2d := ops.generate_op_func(
        op_class=Convolve2D, tensor_only=True, casting=None
    ),
    cross_entropy_loss := ops.generate_op_func(
        op_class=CrossEntropyLoss, tensor_only=True, propagate_kwargs=True, casting=None
    ),
]

__all__ = get_exported_var_names(local_vars=dict(locals()), exported_vars=exported_ops)
