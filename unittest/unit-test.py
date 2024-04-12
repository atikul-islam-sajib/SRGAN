import sys
import unittest
import torch

sys.path.append("src/")

from generator import Generator
from discriminator import Discriminator


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.images = torch.randn(1, 3, 64, 64)
        self.netG = Generator(in_channels=3, out_channels=64)
        self.netD = Discriminator(in_channels=3, out_channels=1)

    def tearDown(self) -> None:
        return super().tearDown()

    def test_netG_shape(self):
        self.assertEqual(
            self.netG(self.images).size(), torch.Size([1, 3, 64 * 4, 64 * 4])
        )

    def test_netD_shape(self):
        self.images = torch.randn(1, 3, 64 * 4, 64 * 4)
        self.assertEquals(self.netD(self.images).size(), torch.Size([1]))


if __name__ == "__main__":
    unittest.main()
