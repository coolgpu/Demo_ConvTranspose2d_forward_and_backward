import torch
from torch import nn
from torch.nn.parameter import Parameter

from my_convTranspose2d_v1_distrib_view import MyConvTranspose2d_v1_Distrib
from my_convTranspose2d_v2_collect_view import MyConvTranspose2d_v2_Collect


def main():
    # set up the test
    nSamples= 1
    inCh = 1
    outCh = 1
    imgSize = 3

    torch.manual_seed(2532)
    A = torch.randint(-5, 10, (nSamples, inCh, imgSize, imgSize), dtype=torch.float64)
    A.requires_grad = True
    A.retain_grad()

    # convTranspose2d arguments
    nPadding = 1
    nStride = 2
    knSize = 3

    input_weight = torch.randint(-3, 3, (nSamples, inCh, imgSize, imgSize), dtype=torch.float64)
    input_bias = torch.zeros(outCh, requires_grad=True, dtype=torch.float64)

    # a copy of data and parameters for testing v1 Distribution View
    B = A.detach().clone()
    B.requires_grad = True
    weightB = input_weight.detach().clone()
    weightB.requires_grad = True
    biasB = input_bias.detach().clone()
    biasB.requires_grad = True

    # a copy of data and parameters for testing v2 Collection View as a typical convolution
    C = A.detach().clone()
    C.requires_grad = True
    weightC = input_weight.detach().clone()
    weightC.requires_grad = True
    biasC = input_bias.detach().clone()
    biasC.requires_grad = True

    # build the ground truth using PyTorch built-in ConvTranspose2d
    # note:     Conv2d.weight shape is [outCh, inCh, Rows, Cols]
    #  ConvTranspose2d.weight shape is [inCh, outCh, Rows, Cols], exchanged the order of inCh and outCh
    torchConvTrans2 = nn.ConvTranspose2d(inCh, outCh, kernel_size=knSize, stride=nStride, padding=nPadding)
    torchConvTrans2.weight = Parameter(input_weight)
    torchConvTrans2.bias = Parameter(input_bias)

    outT = torchConvTrans2(A)
    outT.retain_grad()
    lossT = outT.mean()  # get a scalar value
    lossT.backward()

    # test the implementation #v1
    outB = MyConvTranspose2d_v1_Distrib.apply(B, weightB, biasB, (nPadding, nStride))
    outB.retain_grad()
    lossB = outB.mean()  # get a scalar value to serve as loss
    lossB.backward()

    # test the implementation #v2
    outC = MyConvTranspose2d_v2_Collect(C, weightC, biasC, (nPadding, nStride))
    outC.retain_grad()
    lossC = outC.mean()  # get a scalar value to serve as loss
    lossC.backward()

    # check the results: the input, gradients of the input, weight and bias
    # Compare the custom implementation v1 to the reference
    max_diff_OUTPUT_B_A = (outB - outT).abs().max().detach().numpy()
    diff_grad_X_B_A = B.grad - A.grad
    max_diff_grad_X_B_A = diff_grad_X_B_A.abs().max().detach().numpy()
    diff_grad_w_B_A = weightB.grad - torchConvTrans2.weight.grad
    diff_grad_b_B_A = biasB.grad - torchConvTrans2.bias.grad

    # clean up the infinite small number due to floating point error
    diff_grad_X_B_A[diff_grad_X_B_A < 1e-12] = 0.0
    diff_grad_w_B_A[diff_grad_w_B_A < 1e-12] = 0.0
    diff_grad_b_B_A[diff_grad_b_B_A < 1e-12] = 0.0

    print('\nDifference between v1 and the ground truth:')
    print('max_diff_OUTPUT_B_A: {}'.format(max_diff_OUTPUT_B_A))    
    print('diff_grad_X_B_A max difference: {:.2e}'.format(max_diff_grad_X_B_A))
    if max_diff_grad_X_B_A > 1e-10:
        print('... As a reference, the max value of A.grad is: ', A.grad.abs().max().detach().numpy())
    print('diff_grad_w_B_A:\n', diff_grad_w_B_A.detach().numpy())
    print('diff_grad_b_B_A:\n', diff_grad_b_B_A.detach().numpy())


    # Compare the custom implementation v2 to the reference
    max_diff_OUTPUT_C_A = (outC - outT).abs().max().detach().numpy()
    diff_grad_X_C_A = C.grad - A.grad
    max_diff_grad_X_C_A = diff_grad_X_C_A.abs().max().detach().numpy()
    diff_grad_w_C_A = weightC.grad - torchConvTrans2.weight.grad
    diff_grad_b_C_A = biasC.grad - torchConvTrans2.bias.grad

    # clean up the infinite small number due to floating point error
    diff_grad_X_C_A[diff_grad_X_C_A < 1e-12] = 0.0
    diff_grad_w_C_A[diff_grad_w_C_A < 1e-12] = 0.0
    diff_grad_b_C_A[diff_grad_b_C_A < 1e-12] = 0.0

    print('\nDifference between v2 and the ground truth:')
    print('max_diff_OUTPUT_C_A: {}'.format(max_diff_OUTPUT_C_A))
    print('diff_grad_X_C_A max difference: {:.2e}'.format(max_diff_grad_X_C_A))
    if max_diff_grad_X_C_A > 1e-10:
        print('... As a reference, the max value of A.grad is: ', A.grad.abs().max().detach().numpy())
    print('diff_grad_w_C_A:\n', diff_grad_w_C_A.detach().numpy())
    print('diff_grad_b_C_A:\n', diff_grad_b_C_A.detach().numpy())


    print('Done!')


if __name__=='__main__':
    main()


