### Custom Modules Initialization

Custom modules such as `Loader`, `Trainer`, `Test`, and `Inference` can be initialized and used as follows:

#### Initializing Data Loader
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