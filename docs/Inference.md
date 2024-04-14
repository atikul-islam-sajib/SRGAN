# SR-GAN Inference Tool

This tool performs super-resolution image processing using a pre-trained SR-GAN (Super-Resolution Generative Adversarial Network) model. It is designed to enhance the resolution of images either individually or in batches.

### Command-Line Arguments

| Argument   | Type    | Default | Description                                                |
|------------|---------|---------|------------------------------------------------------------|
| `--image`  | String  | None    | Path to the image file for SRGAN inference.                |
| `--model`  | String  | None    | Path to the trained model file.                            |
| `--device` | String  | "mps"   | The device to perform the inference on (e.g., cpu, cuda, mps). |
| `--single` | Flag    | -       | Perform inference on a single image.                       |
| `--batch`  | Flag    | -       | Perform inference on a batch of images.                    |

### Performing Inference - single + batch(CLI)

```python
python src/inference.py --image "/path/to/image.jpg" --model "/path/to/model.pth" --device "cuda" --single
```

```python
python src/inference.py --model "/path/to/model.pth" --device "cuda" --batch
```

#### Performing Inference - 
```python
inference = Inference(image="path/to/image.jpg", model="path/to/model.pth")
inference.srgan_single()
```

#### Performing Inference - single + batch(Modules)
```python
inference = Inference(model="path/to/model.pth")
inference.srgan_batch()
```

