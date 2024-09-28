# SAM Finetuning with COCO Dataset README

## Overview

This repository provides a PyTorch Lightning implementation for fine-tuning the **Segment Anything Model (SAM)** using **COCO-style segmentation datasets**. It leverages **Segmentation Models PyTorch** for metric computation and a custom COCO dataset loader for mask generation.

The code defines the following key components:
- **`SAMFinetuner`**: A PyTorch Lightning module for fine-tuning SAM.
- **`Coco2MaskDataset`**: A custom dataset class that loads COCO-formatted annotations and generates masks and bounding boxes for training.
- **`all_gather` and `get_world_size`**: Helper functions for distributed training.
  
This setup allows for multi-GPU training with PyTorch's `DistributedDataParallel` (DDP), supports checkpointing, and includes several customizable training options.

## Features

- Fine-tune SAM (`vit_h`, `vit_l`, `vit_b`).
- Distributed training (via PyTorch DDP).
- Training and validation using COCO-formatted datasets.
- Customizable optimizer, learning rate, weight decay, and batch size.
- Supports freezing image encoder, prompt encoder, or mask decoder in the SAM model.
- Implements focal loss, dice loss, and IoU metrics for model evaluation.
- Logging of metrics with intervals.
- Checkpointing and model saving based on validation metrics.

## Requirements

- Python 3.x
- PyTorch 1.9+
- PyTorch Lightning
- torchvision
- segmentation-models-pytorch
- transformers
- OpenCV
- Pillow

## Installation

1. Clone this repository.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Clone the [Segment Anything](https://github.com/facebookresearch/segment-anything) repo into your project folder:
   ```bash
   git clone https://github.com/facebookresearch/segment-anything
   ```

## How to Use

### Command-line Arguments
The script accepts several command-line arguments:

| Argument                 | Description                                 | Default     |
|--------------------------|---------------------------------------------|-------------|
| `--data_root`             | Path to the root directory of the dataset.  | -           |
| `--model_type`            | Type of SAM model (`vit_h`, `vit_l`, `vit_b`) | -           |
| `--checkpoint_path`       | Path to the SAM model checkpoint file.      | -           |
| `--freeze_image_encoder`  | Freeze the image encoder during training.   | False       |
| `--freeze_prompt_encoder` | Freeze the prompt encoder during training.  | False       |
| `--freeze_mask_decoder`   | Freeze the mask decoder during training.    | False       |
| `--batch_size`            | Batch size for training and validation.     | 1           |
| `--image_size`            | Size of the input images (in pixels).       | 1024        |
| `--steps`                 | Number of training steps.                   | 1500        |
| `--learning_rate`         | Learning rate for training.                 | 1e-4        |
| `--weight_decay`          | Weight decay for the optimizer.             | 1e-2        |
| `--metrics_interval`      | Interval for logging metrics.               | 50          |
| `--output_dir`            | Directory to save the model and checkpoints.| `.`         |

### Example Usage

```bash
python train.py --data_root /path/to/coco --model_type vit_h --checkpoint_path /path/to/sam_checkpoint.pth --steps 3000 --batch_size 2 --image_size 512 --learning_rate 1e-4 --output_dir ./outputs
```

### Dataset Structure

The COCO dataset should follow this structure:
```
data_root/
│
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
│
└── val/
    ├── image1.jpg
    ├── image2.jpg
    └── _annotations.coco.json
```

## Training Workflow

1. **Dataset Loading**: The `Coco2MaskDataset` class handles loading of COCO-formatted annotations, generating masks and bounding boxes for each image.
   
2. **Model Initialization**: The SAM model is loaded from a checkpoint and can be partially frozen during training (e.g., image encoder, prompt encoder, mask decoder).

3. **Training**: The model is trained using a combination of **sigmoid focal loss**, **dice loss**, and **IoU loss**. The metrics and losses are logged at specified intervals during training.

4. **Checkpointing**: Checkpoints are saved based on validation metrics, allowing the best-performing model to be retained.

## Model Checkpoints

During training, the model is periodically saved to the `output_dir`, and the top-performing model (based on validation IoU) will be saved with the following naming format:
```
{step}-{val_per_mask_iou:.2f}.ckpt
```

## License

This project uses various components such as SAM and Detectron2, which may have different licenses. Please review the respective licenses before using the code in commercial applications.

## Acknowledgments

This repository integrates components from:
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
