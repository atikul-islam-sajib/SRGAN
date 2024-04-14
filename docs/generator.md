### Generator Class Description

- **Purpose**: Defines the SRGAN Generator model for upscaling low-resolution images to high-resolution images.
- **Components**: Consists of an input block, multiple residual blocks, a middle block, upsample blocks, and an output block.

### Attributes and Parameters

| Attribute/Parameter  | Type              | Description                                                                   |
|----------------------|-------------------|-------------------------------------------------------------------------------|
| `in_channels`        | int               | Number of channels in the input image; typically 3 for RGB.                   |
| `out_channels`       | int               | Number of output channels after the initial block, used throughout the model. |
| `num_repetitive`     | int               | Number of residual blocks to be repeated in the generator.                    |
| `input_block`        | `InputBlock`      | The initial processing block for the input features.                          |
| `residual_block`     | `nn.Sequential`   | Container for multiple instances of `ResidualBlock`.                          |
| `middle_block`       | `MiddleBlock`     | A transition block used before upsampling layers.                             |
| `up_sample`          | `nn.Sequential`   | Container for `UpSampleBlock` instances to upscale features.                  |
| `out_block`          | `OutputBlock`     | The final block to generate the high-resolution output.                       |

### Custom Modules Utilized

- `InputBlock`
- `ResidualBlock`
- `MiddleBlock`
- `UpSampleBlock`
- `OutputBlock`

These modules are designed to handle specific parts of the image processing pipeline in the SRGAN generator, from initial feature extraction to final upsampling and output generation.

### Command-Line Interface (CLI)

The CLI is designed to allow for the testing and parameter adjustment of the `Generator` via command-line arguments.

| Argument         | Type | Default | Description                                  |
|------------------|------|---------|----------------------------------------------|
| `--in_channels`  | int  | 3       | Set the number of input channels.            |
| `--out_channels` | int  | 64      | Set the number of output channels.           |
| `--netG`         | flag | -       | Flag to initiate a test of the generator.    |

### Example Usage

Here's how you can run the generator from the command line, specifying the number of input and output channels and triggering the generation:

```bash
python generator_demo.py --in_channels 3 --out_channels 64 --netG
```