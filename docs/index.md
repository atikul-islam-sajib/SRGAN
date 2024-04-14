# Super-Resolution GAN (SR-GAN) Project

This project provides a complete framework for training and testing a Super-Resolution Generative Adversarial Network (SR-GAN). It includes functionality for data preparation, model training, testing, and inference to enhance low-resolution images to high-resolution.

<img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41524-022-00749-z/MediaObjects/41524_2022_749_Fig1_HTML.png" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized SRGAN model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of SRGAN functionality.                                                         |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Getting Started

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                       |
| ---- | -------------------------------------------- | ------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/SRGAN.git** |
| 2    | Navigate into the project directory.         | **cd SRGAN**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                           |

## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the srgan model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
  - Stores model checkpoints during training for later resumption.
- **best_model/**

  - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**

  - Houses all model checkpoints generated throughout the training process.

- **data/**

  - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
  - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**

  - **Log** files for debugging and tracking model training progress.

- **metrics/**

  - Files related to model performance metrics for evaluation purposes.

- **outputs/**

  - **test_images/**: Images generated during the testing phase, including segmentation outputs.
  - **train_gif/**: GIFs compiled from training images showcasing the model's learning progress.
  - **train_images/**: Images generated during training for performance visualization.

- **research/**

  - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**

  - Source code directory containing all custom modules, scripts, and utility functions for the U-Net model.

- **unittest/**
  - Unit tests ensuring code reliability, correctness, and functionality across various project components.

### Dataset Organization for srgan

The dataset is organized into three categories for SRGAN. Each category directly contains paired images and their corresponding lower resolution images and higher resolution, stored together to simplify the association between lower resolution and higher resolution images .

## Directory Structure:

```
images/
├── Training/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
├── Testing/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
```

### Command Line Interface

The project is controlled via a command line interface (CLI) which allows for running different operational modes such as training, testing, and inference.

#### CLI Arguments
| Argument          | Description                                  | Type   | Default |
|-------------------|----------------------------------------------|--------|---------|
| `--image_path`    | Path to the image dataset                    | str    | None    |
| `--batch_size`    | Number of images per batch                   | int    | 1       |
| `--image_size`    | Size to resize images to                     | int    | 64      |
| `--is_sub_samples`| Whether to subsample the dataset             | bool   | False   |
| `--epochs`        | Number of training epochs                    | int    | 100     |
| `--lr`            | Learning rate                                | float  | 0.0002  |
| `--content_loss`  | Weight of content loss                       | float  | 0.001   |
| `--is_l1`         | Enable L1 regularization                     | bool   | False   |
| `--is_l2`         | Enable L2 regularization                     | bool   | False   |
| `--is_elastic_net`| Enable Elastic Net regularization            | bool   | False   |
| `--is_lr_scheduler`| Enable learning rate scheduler              | bool   | False   |
| `--is_weight_init`| Apply weight initialization                  | bool   | False   |
| `--is_weight_clip`| Apply weight clipping; specify magnitude     | float  | 0.01    |
| `--is_display`    | Display detailed loss information            | bool   | False   |
| `--device`        | Computation device ('cuda', 'mps', 'cpu')    | str    | 'mps'   |
| `--adam`          | Use Adam optimizer                           | bool   | True    |
| `--SGD`           | Use Stochastic Gradient Descent optimizer    | bool   | False   |
| `--beta1`         | Beta1 parameter for Adam optimizer           | float  | 0.5     |
| `--train`         | Flag to initiate training mode               | action | N/A     |
| `--model`         | Path to a saved model for testing            | str    | None    |
| `--test`          | Flag to initiate testing mode                | action | N/A     |
| `--single`        | Flag for single image inference              | action | N/A     |
| `--batch`         | Flag for batch image inference               | action | N/A     |

### CLI Command Examples

| Task                     | Command                                                                                                              |
|--------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | **python main.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --is_l1 True --device "cuda"** |
| **Testing a Model**      | **python main.py --test --model "/path/to/saved_model.pth" --device "cuda"**                                           |
| **Single Image Inference** | **python main.py --single --image "/path/to/image.jpg" --model "/path/to/saved_model.pth" --device "cuda"**            |
| **Batch Image Inference** | **python main.py --batch --model "/path/to/saved_model.pth" --device "cuda"**                                     |n main.py --test --model "path/to/best_model.pth" --device cuda ....
```

### Custom Modules

#### Loader

Initialize and prepare the DataLoader:

```python
loader = Loader(image_path="path/to/images.zip", batch_size=16, image_size=128, is_sub_samples=True)
loader.unzip_folder()
loader.create_dataloader()
```

#### Trainer

Configure and run the training process:

```python
trainer = Trainer(
    epochs=200,
    lr=0.0001,
    content_loss=0.01,
    device='cuda',
    is_display=True,
    is_weight_init=True,
    is_lr_scheduler=True
)
trainer.train()
```

#### Inference

Perform inference with a trained model:

```python
inference = Inference(image="path/to/test_image.jpg", model="path/to/best_model.pth", device='cuda')
inference.srgan_single()  # For single image
inference.srgan_batch()   # For batch processing
```

## Contributing
Contributions to improve this implementation of SRGAN are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).
