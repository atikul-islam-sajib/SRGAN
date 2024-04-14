### Trainer Class Overview

**Purpose:** Manages and conducts training sessions for Generative Adversarial Networks (GANs), specifically handling model setups, training loops, regularization, and learning rate adjustments.

### Attributes and Parameters

| Attribute/Parameter | Type      | Description                                                                              |
|----------------------|-----------|------------------------------------------------------------------------------------------|
| `epochs`             | int       | Total number of epochs to train the models.                                              |
| `lr`                 | float     | Learning rate for the optimizers.                                                        |
| `content_loss`       | float     | Coefficient for the content loss, applied to the generator's output.                     |
| `device`             | str       | Computation device ('cuda', 'mps', 'cpu') where the model will be trained.               |
| `adam`               | bool      | Flag to use Adam optimizer; mutually exclusive with SGD.                                 |
| `SGD`                | bool      | Flag to use Stochastic Gradient Descent optimizer; mutually exclusive with Adam.         |
| `beta1`              | float     | Beta1 hyperparameter for Adam optimizer, affecting the decay rate of the first moment.   |
| `is_l1`              | bool      | Enables L1 regularization if set to True.                                                |
| `is_l2`              | bool      | Enables L2 regularization if set to True.                                                |
| `is_elastic_net`     | bool      | Enables Elastic Net regularization (combination of L1 and L2) if set to True.            |
| `is_lr_scheduler`    | bool      | Enables a learning rate scheduler if set to True.                                        |
| `is_weight_init`     | bool      | Applies custom weight initialization to models if set to True.                           |
| `is_weight_clip`     | bool      | Applies weight clipping to the discriminator during training to stabilize training.       |
| `display`            | bool      | Controls whether to display detailed training progress and loss information.              |

### Custom Modules Utilized

- `InputBlock`
- `FeatureBlock`
- `OutputBlock`
- Various utility functions from `helpers.py` and `utils.py` for loading data, applying transformations, and initializing weights.

### Command-Line Interface (CLI) Options

| Argument             | Type   | Default | Description                                      |
|----------------------|--------|---------|--------------------------------------------------|
| `--epochs`           | int    | 100     | Number of training epochs.                       |
| `--lr`               | float  | 0.0002  | Learning rate for the optimizers.                |
| `--content_loss`     | float  | 0.001   | Multiplier for the content loss.                 |
| `--is_l1`            | bool   | False   | Enable L1 regularization.                        |
| `--is_l2`            | bool   | False   | Enable L2 regularization.                        |
| `--is_elastic_net`   | bool   | False   | Enable Elastic Net regularization.               |
| `--is_lr_scheduler`  | bool   | False   | Activate a learning rate scheduler.              |
| `--is_weight_init`   | bool   | False   | Apply weight initialization.                     |
| `--is_weight_clip`   | float  | 0.01    | Apply weight clipping in netG.                   |
| `--is_display`       | bool   | False   | Display detailed loss information.               |
| `--device`           | str    | "mps"   | Set the computation device.                      |
| `--adam`             | bool   | True    | Use Adam optimizer.                              |
| `--SGD`              | bool   | False   | Use SGD optimizer.                               |
| `--beta1`            | float  | 0.5     | Beta1 hyperparameter for Adam optimizer.         |
| `--train`            | action | -       | Flag to start training the model.                |

### Example Command to Run Training

```python
python train.py --train --epochs 100 --lr 0.0002 --content_loss 0.001 --is_l1 False --is_l2 False --is_elastic_net False --is_lr_scheduler False --is_weight_init False --is_display True --device mps --adam True --SGD False --beta1 0.5
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
