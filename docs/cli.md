## Command-Line Interface (CLI) Arguments

This section details all the command-line arguments available for configuring and running various operations like training, testing, and inference.

### General Arguments
| Argument         | Type   | Default | Description                                             |
|------------------|--------|---------|---------------------------------------------------------|
| `--train`        | Flag   | N/A     | Initiates the training process.                         |
| `--test`         | Flag   | N/A     | Initiates the testing process.                          |
| `--single`       | Flag   | N/A     | Perform inference on a single image.                    |
| `--batch`        | Flag   | N/A     | Perform inference in batch mode.                        |
| `--device`       | String | `"mps"` | Specifies the computation device (e.g., 'cuda', 'mps'). |
| `--model`        | String | None    | Path to the model for testing or inference.             |

### Training Specific Arguments
| Argument              | Type    | Default  | Description                                                        |
|-----------------------|---------|----------|--------------------------------------------------------------------|
| `--image_path`        | String  | None     | Path to the image dataset.                                         |
| `--batch_size`        | Integer | N/A      | Number of images per batch during training.                        |
| `--image_size`        | Integer | N/A      | Size to which the images will be resized.                          |
| `--epochs`            | Integer | `100`    | Number of training epochs.                                         |
| `--lr`                | Float   | `0.0002` | Learning rate for the optimizer.                                   |
| `--content_loss`      | Float   | `1e-3`   | Weight of the content loss in the loss function.                   |
| `--is_l1`             | Boolean | `False`  | Enables L1 regularization if set.                                  |
| `--is_l2`             | Boolean | `False`  | Enables L2 regularization if set.                                  |
| `--is_elastic_net`    | Boolean | `False`  | Enables Elastic Net regularization if set (combines L1 and L2).    |
| `--is_lr_scheduler`   | Boolean | `False`  | Whether to apply a learning rate scheduler.                        |
| `--is_weight_init`    | Boolean | `False`  | Whether to initialize weights before training.                     |
| `--is_weight_clip`    | Float   | `0.01`   | Magnitude of weight clipping for netG, if clipping is applied.     |
| `--is_display`        | Boolean | `False`  | Whether to display detailed training progress and stats.           |
| `--adam`              | Boolean | `True`   | Whether to use the Adam optimizer (mutually exclusive with SGD).   |
| `--SGD`               | Boolean | `False`  | Whether to use Stochastic Gradient Descent (mutually exclusive with Adam). |
| `--beta1`             | Float   | `0.5`    | Beta1 hyperparameter for Adam optimizer.                           |

### Inference Specific Arguments
| Argument        | Type   | Default | Description                                  |
|-----------------|--------|---------|----------------------------------------------|
| `--image`       | String | None    | Path to the image for SRGAN single inference.|


### CLI Command Examples

| Task                     | Command                                                                                                              |
|--------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | **python main.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --is_l1 True --device "cuda"** |
| **Testing a Model**      | **python main.py --test --model "/path/to/saved_model.pth" --device "cuda"**                                           |
| **Single Image Inference** | **python main.py --single --image "/path/to/image.jpg" --model "/path/to/saved_model.pth" --device "cuda"**            |
| **Batch Image Inference** | **python main.py --batch --model "/path/to/saved_model.pth" --device "cuda"**                                     |
