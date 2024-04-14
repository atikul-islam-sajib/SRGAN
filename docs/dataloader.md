### DataLoader

## Parameters for Loader Class

| Parameter        | Type    | Default | Description                                                                                                                                                 |
|------------------|---------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `image_path`     | str     | None    | Specifies the path to a zip file containing the images. This zip file will be processed to extract images for training and testing.                           |
| `in_channels`    | int     | 3       | Indicates the number of channels in the input images, typical for color images (RGB) which have three channels.                                              |
| `batch_size`     | int     | 1       | Determines the number of images to be processed in a single batch. This affects both memory usage and training speed.                                        |
| `image_size`     | int     | 64      | Defines the size to which each image will be resized. This uniform size is necessary for processing through the neural network.                              |
| `is_sub_samples` | bool    | False   | A flag to determine whether the dataset should be subsampled to a smaller size, useful for quick tests or debugging with reduced computational requirements. |

### Method Descriptions for Loader Class

In addition to the parameters, it's helpful to outline key methods within the `Loader` class:

| Method                   | Description                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------------------|
| `image_transforms()`     | Sets up the transformations to be applied to each image, including resizing, tensor conversion, and normalization based on the given parameters. |
| `unzip_folder()`         | Extracts images from a provided zip file to a specified directory for further processing.         |
| `process_images()`       | Applies the defined image transformations to preprocess the images before they are passed to the neural network. |
| `get_subsample_of_dataloader()` | Optionally reduces the size of the training and testing datasets based on the `is_sub_samples` flag. |
| `extract_images()`       | Handles the extraction and preprocessing of images from the raw dataset directory into structured training and testing datasets. |
| `create_dataloader()`    | Initializes PyTorch DataLoader objects with the processed datasets, ready for use in training or testing phases. |

### Usage Example in Code

Below is a sample Python script snippet that demonstrates initializing and using the `Loader` class:

```python
from loader import Loader

# Initialize the Loader with the desired configuration
image_loader = Loader(
    image_path="path/to/your/image_data.zip",
    in_channels=3,
    batch_size=32,
    image_size=128,
    is_sub_samples=True
)

# Process the images and create data loaders
image_loader.unzip_folder()
image_loader.create_dataloader()

# Optionally, display the processed images
image_loader.display_images()
```

# SRGAN Loader Class - CLI

## Overview
This README document provides detailed information about the command-line arguments and parameters for the SRGAN Loader class. The Loader class is designed for processing and creating data loaders for SRGAN training and testing datasets.

### Command-Line Arguments

| Argument          | Type    | Default | Description                                             |
|-------------------|---------|---------|---------------------------------------------------------|
| `--image_path`    | str     | None    | Path to the zip file containing raw images.             |
| `--batch_size`    | int     | 1       | Batch size for the DataLoader.                          |
| `--image_size`    | int     | 64      | Size to which input images are resized.                 |
| `--is_sub_samples`| bool    | False   | Whether to create a subsample of the dataset for tests. |

## Usage Examples

### Initializing and Running the Loader
To initialize the loader with specific parameters and process images:

```bash
python your_script.py --image_path "./path/to/images.zip" --batch_size 32 --image_size 128 --is_sub_samples True
```

### Help Command
To view all available options and their descriptions, you can use the help command:

```bash
python your_script.py --help
```
