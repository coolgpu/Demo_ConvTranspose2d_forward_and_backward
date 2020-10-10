"""
Online lecture: Basics of PyTorch autograd
Demonstrate custom implementation #1 of forward and backward propagation of ConvTranspose2d
"""
import torch
from my_conv2d_v1 import MyConv2d_v1
from my_conv2d_v2 import MyConv2d_v2


def MyConvTranspose2d_v1(Y, in_weight, in_bias=None, convparam=None):
    # Note: both padding and stride are those used in the original forward Conv2d, ie. from X --> Y
    if convparam is not None:
        padding, stride = convparam
    else:
        padding, stride = 0, 1

    # for simplicity, let's consider only square kernels: nKnRows = nKnCols
    nInYCh, nOutXCh, nKnRows, nKnCols = in_weight.shape
    nImgSamples, nInYCh, nYRows, nYCols = Y.shape

    # treat convTranspose2d as a new convolution, in which
    # newPadding = nKn - 1 - padding
    # newStride = 1
    # newY is a new array with a new size of
    # (nImgSamples, nInYCh, nNewYRows, nNewYCols) with (stride-1) zeros inserted between
    # adjacent rows (or columns), where
    # nNewYRows = (nYRows-1)*stride + 1 and nNewYCols = (nYCols-1)*stride + 1
    # new_in_weight = flip of the original in_weight

    newPadding = nKnRows - 1 - padding
    nNewYRows = (nYRows - 1) * stride + 1
    nNewYCols = (nYCols - 1) * stride + 1
    newY = torch.zeros((nImgSamples, nInYCh, nNewYRows, nNewYCols), dtype=Y.dtype)
    newY[:, :, 0:nNewYRows:stride, 0:nNewYCols:stride] = Y
    new_in_weight = torch.flip(in_weight, [2, 3]).transpose(0, 1)
    # outX = MyConv2d_v1.apply(newY, new_in_weight, in_bias, (newPadding, 1))
    outX = MyConv2d_v2.apply(newY, new_in_weight, in_bias, (newPadding, 1))

    return outX
