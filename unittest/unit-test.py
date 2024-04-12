import sys
import unittest
import torch

sys.path.append("src/")

from generator import Generator
from discriminator import Discriminator
from feature_extractor import VGG16


class UnitTest(unittest.TestCase):
    """
    Unit tests for the Generator and Discriminator models within a GAN architecture.

    This test suite evaluates:
    - Output shape correctness for both models.
    - Total number of parameters in each model to ensure model structure consistency.
    """

    def setUp(self):
        """
        Set up function to initialize the test environment, instantiating the generator and discriminator
        models with predefined input and output channel specifications, along with a sample image tensor.
        """
        self.images = torch.randn(1, 3, 64, 64)
        self.netG = Generator(in_channels=3, out_channels=64)
        self.netD = Discriminator(in_channels=3, out_channels=64)
        self.vgg = VGG16(pretrained=True)

    def tearDown(self) -> None:
        """
        Tear down any created data structures after tests are completed. Not used here but included for future use.
        """
        super().tearDown()

    def test_netG_shape(self):
        """
        Test to verify that the output shape from the generator model matches expected dimensions.
        """
        expected_size = torch.Size(
            [1, 3, 64 * 4, 64 * 4]
        )  # 64 * 4 due to upsampling in generator
        self.assertEqual(self.netG(self.images).size(), expected_size)

    def test_netD_shape(self):
        """
        Test to verify that the output shape from the discriminator model is correct.
        Specifically, the discriminator should output a single value per input.
        """
        self.images = torch.randn(
            1, 3, 64 * 4, 64 * 4
        )  # Adjust size to expected input dimensions for discriminator
        self.assertEqual(self.netD(self.images).size(), torch.Size([1]))

    def test_netG_total_params(self):
        """
        Test to verify that the total number of trainable parameters in the generator is as expected.
        """
        total_params = sum(params.numel() for params in self.netG.parameters())
        self.assertEqual(total_params, 1549461)  # Pre-calculated expected value

    def test_netD_total_params(self):
        """
        Test to verify that the total number of trainable parameters in the discriminator is as expected.
        """
        total_params = sum(params.numel() for params in self.netD.parameters())
        self.assertEqual(total_params, 5252481)  # Pre-calculated expected value

    def test_VGG16_shape(self):
        """
        Test to verify that the total number of trainable parameters in the discriminator is as expected.
        """
        self.assertEquals(self.vgg(self.images).size(), torch.Size([1, 256, 16, 16]))


if __name__ == "__main__":
    unittest.main()
