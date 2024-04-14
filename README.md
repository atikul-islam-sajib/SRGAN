# Super-Resolution GAN (SR-GAN) Project

<img src="https://raw.githubusercontent.com/atikul-islam-sajib/Research-Assistant-Work-/main/IMG_9292.jpg" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

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

For detailed documentation on the dataset visit the [Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

### Documentation SRGAN

For detailed documentation on the implementation and usage, visit the -> [SRGAN Documentation](https://atikul-islam-sajib.github.io/SRGAN-deploy/).

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

| Task                     | CUDA Command                                                                                                              | MPS Command                                                                                                              | CPU Command                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | `python main.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --is_l1 True --device "cuda"` | `python main.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --is_l1 True --device "mps"` | `python main.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --is_l1 True --device "cpu"` |
| **Testing a Model**      | `python main.py --test --model "/path/to/saved_model.pth" --device "cuda"`                                              | `python main.py --test --model "/path/to/saved_model.pth" --device "mps"`                                              | `python main.py --test --model "/path/to/saved_model.pth" --device "cpu"`                                              |
| **Single Image Inference** | `python main.py --single --image "/path/to/image.jpg" --model "/path/to/saved_model.pth" --device "cuda"`               | `python main.py --single --image "/path/to/image.jpg" --model "/path/to/saved_model.pth" --device "mps"`               | `python main.py --single --image "/path/to/image.jpg" --model "/path/to/saved_model.pth" --device "cpu"`               |
| **Batch Image Inference** | `python main.py --batch --model "/path/to/saved_model.pth" --device "cuda"`                                           | `python main.py --batch --model "/path/to/saved_model.pth" --device "mps"`                                           | `python main.py --batch --model "/path/to/saved_model.pth" --device "cpu"`                                           |

### Notes:
- **CUDA Command**: For systems with NVIDIA GPUs, using the `cuda` device will leverage GPU acceleration.
- **MPS Command**: For Apple Silicon (M1, M2 chips), using the `mps` device can provide optimized performance.
- **CPU Command**: Suitable for systems without dedicated GPU support or for testing purposes on any machine.


#### Initializing Data Loader - Custom Modules
```python
loader = Loader(image_path="path/to/dataset", batch_size=32, image_size=128)
loader.unzip_folder()
loader.create_dataloader()
```

##### To details about dataset
```python
print(loader.details_dataset())   # It will give a CSV file about dataset
loader.display_images()           # It will display the images from dataset
```

#### Training the Model
```python
trainer = Trainer(
    epochs=100,                # Number of epochs to train the model
    lr=0.0002,                 # Learning rate for optimizer
    content_loss=0.1,          # Weight for content loss in the loss calculation
    device='cuda',             # Computation device ('cuda', 'mps', 'cpu')
    adam=True,                 # Use Adam optimizer; set to False to use SGD if implemented
    SGD=False,                 # Use Stochastic Gradient Descent optimizer; typically False if Adam is True
    beta1=0.5,                 # Beta1 parameter for Adam optimizer
    is_l1=False,               # Enable L1 regularization
    is_l2=False,               # Enable L2 regularization
    is_elastic_net=False,      # Enable Elastic Net regularization (combination of L1 and L2)
    is_lr_scheduler=False,     # Enable a learning rate scheduler
    is_weight_init=False,      # Enable custom weight initialization for the models
    is_weight_clip=False,      # Enable weight clipping within the training loop; use a float value for clip magnitude if True
    is_display=True            # Display training progress and statistics
)

# Start training
trainer.train()
```

##### Training Performances
```python
print(trainer.plot_history())    # It will plot the netD and netG losses for each epochs
```

#### Testing the Model
```python
test = Test(device="cuda", model="path/to/model.pth") # use mps, cpu
test.plot()
```

#### Performing Inference
```python
inference = Inference(image="path/to/image.jpg", model="path/to/model.pth")
inference.srgan_single()
```

#### Performing Inference - batch
```python
inference = Inference(model="path/to/model.pth")
inference.srgan_batch()
```

## Contributing
Contributions to improve this implementation of SRGAN are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).
