# import pandas as pd

# print(
#     pd.DataFrame(
#         {
#             "Train": [1200],
#             "test": [1000],
#         },
#         index=["quantity", "quan"],
#     ).transpose()
# )


import torch
import sys
import matplotlib.pyplot as plt

sys.path.append("src/")

from generator import Generator

from utils import load

data = load(
    filename="/Users/shahmuhammadraditrahman/Desktop/SRGAN/data/processed/train_dataloader.pkl"
)

lr, hr = next(iter(data))

load_state = torch.load(
    "/Users/shahmuhammadraditrahman/Desktop/SRGAN/checkpoints/train_models/netG49.pth"
)

# print(load_state["netG"])

netG = Generator()
netG.load_state_dict(load_state["netG"])

images = netG(lr)

lr1 = lr[0].permute(1, 2, 0).squeeze().detach().numpy

# print(lr.shape)

image = images[0].permute(1, 2, 0).squeeze().detach().numpy()
image = (image - image.min()) / (image.max() - image.min())

print(image.shape)

plt.imshow(image, cmap="gray")
plt.show()
