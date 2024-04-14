## Neural Network Blocks Parameters and Command-Line Arguments

| Block Name     | Parameter      | Type | Default | Description                                                       | Command-Line Argument | Example Value |
|----------------|----------------|------|---------|-------------------------------------------------------------------|----------------------|---------------|
| **InputBlock** | `in_channels`  | int  | None    | Number of input channels.                                         | `--in_channels`      | 3             |
|                | `out_channels` | int  | None    | Number of output channels.                                        | `--out_channels`     | 64            |
| **MiddleBlock**| `in_channels`  | int  | None    | Number of input channels.                                         | `--in_channels`      | 64            |
|                | `out_channels` | int  | None    | Number of output channels.                                        | `--out_channels`     | 128           |
| **OutputBlock**| `in_channels`  | int  | None    | Number of input channels to the block.                            | `--in_channels`      | 64            |
|                | `out_channels` | int  | None    | Desired number of output channels.                                | `--out_channels`     | 3             |
| **ResidualBlock**| `in_channels`| int  | None    | Number of input channels.                                         | `--in_channels`      | 64            |
|                | `out_channels` | int  | None    | Number of output channels.                                        | `--out_channels`     | 64            |
| **UpSampleBlock**| `in_channels`| int  | None    | Number of input channels.                                         | `--in_channels`      | 64            |
|                | `out_channels` | int  | None    | Number of output channels; should be a multiple of upscale factor.| `--out_channels`     | 256           |

### 1. InputBlock

**Description**:
- Initializes with specified input and output channels, setting up a convolutional layer followed by a PReLU activation based on the `params()` function.

**Command-Line Parameters**:
- `--in_channels`: Specifies the number of channels in the input images.
- `--out_channels`: Specifies the number of output channels from the convolutional layer.

**Example Command**:
```bash
python input_block_demo.py --in_channels 3 --out_channels 64
```

### 2. MiddleBlock

**Description**:
- Implements a middle block for the generator network, using convolutional layers and batch normalization. It supports skip connections, enabling addition of input with another tensor before subsequent layers.

**Command-Line Parameters**:
- `--in_channels`: Specifies the number of input channels.
- `--out_channels`: Specifies the number of output channels.

**Example Command**:
```bash
python middle_block_demo.py --in_channels 64 --out_channels 128
```

### 3. OutputBlock

**Description**:
- Applies a convolutional layer followed by a Tanh activation to generate the final output of the generator network.

**Command-Line Parameters**:
- `--in_channels`: Specifies the number of input channels to the block.
- `--out_channels`: Specifies the desired number of output channels.

**Example Command**:
```bash
python output_block_demo.py --in_channels 64 --out_channels 3
```

### 4. ResidualBlock

**Description**:
- A residual block designed to prevent the vanishing gradient problem by allowing an alternate shortcut path for the gradient. Includes two convolutional layers followed by batch normalization and PReLU activation.

**Command-Line Parameters**:
- `--in_channels`: Specifies the number of input channels.
- `--out_channels`: Specifies the number of output channels.

**Example Command**:
```bash
python residual_block_demo.py --in_channels 64 --out_channels 64
```

### 5. UpSampleBlock

**Description**:
- Increases the spatial dimensions of input feature maps using a convolution followed by a PixelShuffle operation.

**Command-Line Parameters**:
- `--in_channels`: Specifies the number of input channels.
- `--out_channels`: Specifies the number of output channels, which should be a multiple of the square of the upscale factor.

**Example Command**:
```bash
python upsample_block_demo.py --in_channels 64 --out_channels 256
```
