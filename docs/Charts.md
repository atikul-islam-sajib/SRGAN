### Neural Network Testing Tool

This tool is designed for testing and evaluating neural network models, specifically focusing on image generation tasks using a pre-trained generator model. It includes functionalities for evaluating the model on a test dataset, visualizing the results, and generating GIFs to visualize the model's performance over time.

## Features

- **Model Testing:** Load and test neural networks, particularly generator models.
- **Image Visualization:** Generate and compare low-resolution, high-resolution, and super-resolved images side by side.
- **Best Model Selection:** Automatically select the best model based on performance metrics.
- **GIF Creation:** Compile a series of images into an animated GIF to showcase the model's output over time.

### Command Line Interface (CLI) Commands

| Command      | Description                                      | Example                              |
|--------------|--------------------------------------------------|--------------------------------------|
| `--device`   | Specify the computation device ('cuda', 'mps', 'cpu'). | `--device cuda`                      |
| `--model`    | Path to a pre-trained model to load and test.    | `--model path/to/model.pth`          |
| `--test`     | Trigger to start the testing process.             | `--test`                             |

### Example Command - CLI

Run the testing process on a CUDA device with a specified model:

```bash
python test.py --device cuda --model path/to/model.pth --test
```

### Example Command -  Modules

```python
test = Test(device="cuda", model="path/to/model.pth") # use mps, cpu
test.plot()
```

### Parameters and Arguments

| Parameter        | Type    | Default | Description                                                    |
|------------------|---------|---------|----------------------------------------------------------------|
| `device`         | string  | "mps"   | The computation device (e.g., 'cuda', 'mps', 'cpu').           |
| `model`          | string  | None    | Optional. Path to a pre-trained model to load and test.        |
| `test`           | boolean | False   | Flag to trigger the testing process.                           |

## Testing and Visualization

Upon executing the test command, the script performs the following operations:

1. **Loads the dataset:** Fetches the test data from the `PROCESSED_DATA_PATH`.
2. **Model Evaluation:** The provided or newly initialized model is used to generate images based on the test dataset.
3. **Visual Comparison:** Plots low-resolution input images alongside the model's high-resolution outputs and actual high-resolution images for comparison.
4. **GIF Creation:** If there are multiple images in the `TRAIN_IMAGES` directory, it compiles them into an animated GIF to visually depict the model's performance across different stages or epochs.
