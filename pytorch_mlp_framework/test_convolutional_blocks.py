import torch
import unittest

from pytorch_mlp_framework.model_architectures import (
    ConvolutionalProcessingBlockBN,
    ConvolutionalDimensionalityReductionBlockBN,
    ConvolutionalProcessingBlockBNRes,
    ConvolutionalDimensionalityReductionBlockBNRes
)

class TestConvolutionalBlocks(unittest.TestCase):
    def test_processing_block_bn(self):
        input_shape = (1, 3, 32, 32)
        x = torch.randn(input_shape)
        block = ConvolutionalProcessingBlockBN(
            input_shape=input_shape,
            num_filters=16,
            kernel_size=3,
            padding=1,
            bias=False,
            dilation=1
        )
        with torch.no_grad():
            output = block(x)

        expected_shape = (1, 16, 32, 32)
        self.assertEqual(output.shape, torch.Size(expected_shape))

    def test_dim_reduction_block_bn(self):
        input_shape = (1, 16, 32, 32)
        x = torch.randn(input_shape)
        block = ConvolutionalDimensionalityReductionBlockBN(
            input_shape=input_shape,
            num_filters=32,
            kernel_size=3,
            padding=1,
            bias=False,
            dilation=1,
            reduction_factor=2
        )
        with torch.no_grad():
            output = block(x)
        expected_shape = (1, 32, 16, 16)
        self.assertEqual(output.shape, torch.Size(expected_shape))

    def test_processing_block_bn_res(self):
        input_shape = (1, 16, 16, 16)
        x = torch.randn(input_shape)
        block = ConvolutionalProcessingBlockBNRes(
            input_shape=input_shape,
            num_filters=16,
            kernel_size=3,
            padding=1,
            bias=False,
            dilation=1
        )
        with torch.no_grad():
            output = block(x)
        expected_shape = (1, 16, 16, 16)
        self.assertEqual(output.shape, torch.Size(expected_shape))

    def test_dim_reduction_block_bn_res(self):
        input_shape = (1, 32, 16, 16)
        x = torch.randn(input_shape)
        block = ConvolutionalDimensionalityReductionBlockBNRes(
            input_shape=input_shape,
            num_filters=64,
            kernel_size=3,
            padding=1,
            bias=False,
            dilation=1,
            reduction_factor=2
        )
        with torch.no_grad():
            output = block(x)
        expected_shape = (1, 64, 8, 8)
        self.assertEqual(output.shape, torch.Size(expected_shape))

if __name__ == '__main__':
    unittest.main()
