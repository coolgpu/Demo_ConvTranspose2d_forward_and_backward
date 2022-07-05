"""
Online lecture: Basics of PyTorch autograd
Demonstrate custom implementation version #1 of forward and backward propagation of ConvTranspose2d
"""
import torch
from torch.autograd import Function


class MyConvTranspose2d_v1_Distrib(Function):
    """
    Version #1 of our own custom autograd Functions of MyConvTranspose2d by subclassing
    torch.autograd.Function and overrdie the forward and backward passes
    Version #1 is based on the Distribution perspective view
    """
    @staticmethod
    def forward(ctx, Y, in_weight, in_bias=None, convparam=None):
        """ override the forward function """
        # note: for demo purpose, assume dilation=1 and padding_mode='zeros',
        # Note: assume the padding and stride is the same for ROWS and COLS, respectively
        # Note: both padding and stride are those used in the original forward Conv2d, ie. from X --> Y
        if convparam is not None:
            padding, stride = convparam
        else:
            padding, stride = 0, 1

        nInYCh, nOutXCh, nKnRows, nKnCols = in_weight.shape
        nImgSamples, nInYCh, nYRows, nYCols = Y.shape

        # Determine the output shape without considering user specified output size
        # Note that, for non-unit stride in forward Convolution, there are multiple (=STRIDE) different 
        # ORIGINAL INPUT sizes that lead to the same size of this Y (Y is output from the forward convolution).
        # One of those ORIGINAL INPUT sizes satisfies (OriginalSize+2P-K) % Stride == 0 while the other
        # (STRIDE-1) ORIGINAL INPUT sizes don't.
        # Let's first consider the one that satisfies (OriginalSize+2P-K) % Stride == 0.
        # The following 4 nOut dimensions are for the original X (that is convolved to generate this Y)
        nOutXRows = (nYRows-1)*stride - 2 * padding + nKnRows
        nOutXCols = (nYCols-1)*stride - 2 * padding + nKnCols
        nOutXRowsPadded = nOutXRows + 2 * padding
        nOutXColsPadded = nOutXCols + 2 * padding

        paddedOrigX = torch.zeros((nImgSamples, nOutXCh, nOutXRowsPadded, nOutXColsPadded), dtype=Y.dtype)

        for inYCh in range(nInYCh):
            for iYRow in range(nYRows):
                startRow = iYRow * stride
                for iYCol in range(nYCols):
                    startCol = iYCol * stride

                    paddedOrigX[:, :, startRow:startRow+nKnRows, startCol:startCol+nKnCols] += \
                        Y[:, inYCh, iYRow, iYCol].reshape(-1, 1, 1, 1) * \
                        in_weight[inYCh, :, 0:nKnRows, 0:nKnCols]

        outOrigX = paddedOrigX[:, :, padding:nOutXRows+padding, padding:nOutXCols+padding]

        if in_bias is not None:
            outOrigX += in_bias.view(1, -1, 1, 1)

        ctx.parameters = (padding, stride)
        ctx.save_for_backward(Y, in_weight, in_bias)

        return outOrigX

    @staticmethod
    def backward(ctx, grad_from_upstream):
        """
        override the backward function. It receives a Tensor containing the gradient of the loss
        with respect to the output of the custom forward pass, and calculates the gradients of 
        the loss with respect to each of the inputs of the custom forward pass.
        """
        print('Performing custom backward of MyConvTranspose2d_v1_Distrib')
        padding, stride = ctx.parameters
        Y, in_weight, in_bias = ctx.saved_tensors
        nInYCh, nOutXCh, nKnRows, nKnCols = in_weight.shape
        nImgSamples, nInYCh, nYRows, nYCols = Y.shape
        nImgSamples, nOutXCh, nOutXRows, nOutXCols = grad_from_upstream.shape
        grad_upstream_padded = torch.zeros(nImgSamples, nOutXCh, nOutXRows+2*padding, nOutXCols+2*padding)
        grad_upstream_padded[:, :, padding:nOutXRows+padding, padding:nOutXCols+padding] = grad_from_upstream

        grad_Y = torch.zeros_like(Y)
        grad_weight = torch.zeros_like(in_weight)

        for inYCh in range(nInYCh):
            for iYRow in range(nYRows):
                startRow = iYRow * stride
                for iYCol in range(nYCols):
                    startCol = iYCol * stride

                    grad_Y[:, inYCh, iYRow, iYCol] = \
                        (grad_upstream_padded[:, :, startRow:startRow+nKnRows, startCol:startCol+nKnCols] *
                        in_weight[inYCh, :, 0:nKnRows, 0:nKnCols]).sum(axis=(1, 2, 3))

                    grad_weight[inYCh, :, 0:nKnRows, 0:nKnCols] += \
                        (grad_upstream_padded[:, :, startRow:startRow+nKnRows, startCol:startCol+nKnCols] *
                        Y[:, inYCh, iYRow, iYCol].reshape(-1, 1, 1, 1)).sum(axis=0)

        if in_bias is not None:
            grad_bias = grad_from_upstream.sum(axis=(0, 2, 3))
        else:
            grad_bias = None

        return grad_Y, grad_weight, grad_bias, None
