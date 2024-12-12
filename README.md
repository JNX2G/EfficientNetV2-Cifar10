# EfficientNetV2-Cifar10
This repository contains PyTorch code for training and evaluating an EfficientNetV2 model on the CIFAR-10 dataset. It includes features like mixed precision training, gradient accumulation, and exponential moving average (EMA) for better performance and efficient resource utilization.

## Requirements
- Python 3.10.14
- pytorch 2.4.0+cu121
- wandb
  
## Features
- **Dataset**: CIFAR-10, preprocessed with data augmentation and normalization.
- **Model**: Pre-trained [EfficientNetV2](https://github.com/hankyul2/EfficientNetV2-pytorch) loaded from PyTorch Hub.
  - efficientnet_v2_s_in21k
- **Techniques**:
  - Mixed precision training using `torch.cuda.amp` for faster training.
  - Gradient accumulation to simulate larger batch sizes.
  - OneCycle Learning Rate Scheduler.
  - Exponential Moving Average (EMA) for stable training.
- **Hardware**: Optimized for GPU with CUDA.
