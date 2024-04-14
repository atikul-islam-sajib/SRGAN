## Overview of Blocks

---

## Parameters and Command-Line Arguments

| Class          | Parameters                          | Command-Line Arguments                              | Description                                                  |
|----------------|-------------------------------------|-----------------------------------------------------|--------------------------------------------------------------|
| `FeatureBlock` | in_channels, out_channels,          | --in_channels, --out_channels                       | Configures the number of input and output channels, kernel size, stride, and padding. |
|                | kernel_size, stride, padding        |                                                     |                                                              |
| `InputBlock`   | in_channels, out_channels           | --in_channels, --out_channels                       | Sets the input and output channels and fetches convolution parameters from `params()`. |
| `OutputBlock`  | in_channels, out_channels           | --in_channels, --out_channels                       | Specifies input and output channels, with the kernel size matching the output channels. |

---

### FeatureBlock
- **Purpose**: Extracts features using convolution, batch normalization, and LeakyReLU activation.
- **Usage**: Typically used in the early layers of the discriminator.

### InputBlock
- **Purpose**: Processes initial input images using convolutions and activations to prepare features.
- **Usage**: Acts as the entry point for the discriminator network.

### OutputBlock
- **Purpose**: Consolidates features to output a final decision value.
- **Usage**: Used as the final layer in the discriminator to determine the authenticity of the input.


## Examples

### FeatureBlock Example
```python
feature_block = FeatureBlock(in_channels=3, out_channels=64)
images = torch.randn(1, 3, 256, 256)
output = feature_block(images)
print(output.size())  # Expected size: [1, 64, 128, 128]
```

### InputBlock Example
```python
input_block = InputBlock(in_channels=3, out_channels=64)
images = torch.randn(1, 3, 64, 64)
output = input_block(images)
print(output.size())  # Expected size: [1, 64, 16, 16]
```

### OutputBlock Example
```python
output_block = OutputBlock(in_channels=64, out_channels=64)
images = torch.randn(1, 512, 4, 4)
output = output_block(images)
print(output.size())  # Expected size: [1, 1, 1, 1]
```

---

## Command-Line Interface

Use the command-line interface to specify parameters for each block during testing or demonstration. For instance:

```bash
python feature_block_demo.py --in_channels 3 --out_channels 64
python input_block_demo.py --in_channels 3 --out_channels 64
python output_block_demo.py --in_channels 64 --out_channels 64
```