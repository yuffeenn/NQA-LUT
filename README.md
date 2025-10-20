# NQA-LUT

Software implementation of "Neural Quantization-Aware Piecewise Linear Approximation for Nonlinear Functions"

## Overview

The project provides a two-stage training pipeline: pre-training + quantization-aware fine-tuning:
- **ReluNN**: A neural network model for learning nonlinear functions
- **QPWL**: Quantized Piecewise Linear model for efficient inference

## Project Structure

```
├─config/                 # Configuration files
├─log/                    # Training logs and checkpoints
├─table1/                 # Experiments for Table 1
├─table2/                 # Experiments for Table 2
├─parse_results.py        # Results parsing utilities
├─relunn.py               # ReluNN model implementation
├─ppwl.py                 # ReluNN with dirct PTQ
├─qpwl.py                 # Quantized Piecewise Linear Approximation
├─train.py                # Main training script
├─utils.py                # Utility functions
├─LICENSE                 # License file
└─README.md               # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yuffeenn/NQA-LUT
cd NQA-LUT
# Install dependencies
pip install torch pyyaml tabulate
```
## Usage

### Configuration

Create a YAML configuration file like:

```yaml
func_name: gelu                    # Target function to approximate
criterion_name: 'mae'              # Loss metric (MAE or MSE)
num_entries: 8                     # Number of piecewise segments
x_range: [-4.0, 2.0]               # Input range for approximation
train: True                        # Enable pre-training
train_epochs: 40000                # Pre-training epochs
train_lr: 1.0e-3                   # Pre-training learning rate
weight_decay: 1.0e-4               # Weight decay
range_weight: 0.1                  # Range constraint weight
prox_weight: 0.1                   # Proximity constraint weight
qat: True                          # Enable quantization-aware training
bits: 8                            # Quantization bit-width
qat_epochs: 5000                   # QAT training epochs
qat_lr: 1.0e-3                     # QAT learning rate
```

### Basic Training

```python
python train.py --config config/gelu.yaml
```
### Outputs

Training generates:
- Model checkpoints in `log/` directory
- Training logs and metrics
- Visualization of the PWL

## Results

### Table 1: MAE Across Nonlinear Activation Functions

| Function |  Range  | Entry | Bits |               MAE               |
| :------- | :-----: | :---: | :--: | :-----------------------------: |
| Sigmoid  | [-8, 8] |  10   |  10  | [1.63e⁻³](./table1/log/sigmoid) |
| GELU     | [-4, 2] |   7   |  11  |  [2.71e⁻³](./table1/log/gelu)   |
| ELU      | [-6, 2] |  10   |  9   |   [1.59e⁻³](./table1/log/elu)   |
| SiLU     | [-8, 3] |  10   |  11  |  [3.41e⁻³](./table1/log/silu)   |
| Tanh     | [-5, 5] |  10   |  10  |  [3.23e⁻³](./table1/log/tanh)   |

### Table 2: MSE for Nonlinear Operations in Transformers (INT8)

| Operations |   Range   |                 8-Entry                  |                 16-Entry                  |
| :--------- | :-------: | :--------------------------------------: | :---------------------------------------: |
| GELU       |  [-4, 4]  |  [7.0e⁻⁵](./table2/log_e8/gelu)  |  [7.7e⁻⁶](./table2/log_e16/gelu)  |
| HardSwish  |  [-4, 4]  | [1.2e⁻⁴](./table2/log_e8/hswish) | [1.4e⁻⁵](./table2/log_e16/hswish) |
| Exp        |  [-8, 0]  |  [6.5e⁻⁶](./table2/log_e8/exp)   |  [3.1e⁻⁶](./table2/log_e16/exp)   |
| Div        | [0.5, 4]  |  [2.5e⁻⁵](./table2/log_e8/div)   |  [1.2e⁻⁵](./table2/log_e16/div)   |
| Rsqrt      | [0.25, 4] | [2.8e⁻⁵](./table2/log_e8/rsqrt)  | [6.3e⁻⁶](./table2/log_e16/rsqrt)  |
