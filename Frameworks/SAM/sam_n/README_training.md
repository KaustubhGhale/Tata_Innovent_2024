Here’s a `README.md` file for the provided script:

---

# SAM Model Training Pipeline

This repository contains a training pipeline for a Segment Anything Model (SAM) using a dataset of images and masks. The SAM model is adapted from Facebook's pre-trained SAM model, and is fine-tuned on a custom dataset for semantic segmentation tasks.


## Requirements

To run this code, the following libraries are required:

- `numpy`
- `Pillow`
- `torch`
- `monai`
- `tqdm`
- `transformers`
- `datasets`
- `patchify`
- `matplotlib`

Install the dependencies using the following command:

```bash
pip install numpy pillow torch monai tqdm transformers datasets patchify matplotlib
```

## Dataset Structure

The dataset is assumed to be organized as follows:

```
Dataset/
├── train/
│   ├── images/     # Directory containing the training images (PNG, JPG, JPEG)
│   ├── masks/      # Directory containing the corresponding masks (PNG, JPG, JPEG)
```

- **Images**: RGB images of any size.
- **Masks**: Grayscale masks where the segmented regions are represented by non-zero values (normalized to [0, 1]).

The images and masks are processed into patches of size 256x256 pixels using the `patchify` library, and only non-empty patches (those with non-zero masks) are used for training.

## Hyperparameters

The following hyperparameters are set in the script:

| Hyperparameter   | Value      | Description                                 |
|------------------|------------|---------------------------------------------|
| `patch_size`     | `256`      | Size of the patches extracted from images.  |
| `step`           | `256`      | Step size for patch extraction.             |
| `batch_size`     | `2`        | Batch size for DataLoader.                  |
| `learning_rate`  | `1e-5`     | Learning rate for the optimizer.            |
| `weight_decay`   | `0`        | Weight decay for the optimizer.             |
| `num_epochs`     | `5`        | Number of training epochs.                  |
| `loss_fn_type`   | `DiceCELoss`| Type of loss function used for training.    |
| `model_save_path`| Path to save the trained model (`.pth` file). |

## Usage

1. Clone the repository.
2. Set the paths to your dataset and model saving directory.
3. Ensure that the necessary libraries are installed.
4. Run the script using the following command:

```bash
python train_sam.py
```

### Example Paths

Set your dataset and model paths in the script:

```python
image_dir = r"/path/to/images"
mask_dir = r"/path/to/masks"
model_save_path = r"/path/to/save/model.pth"
```

## Model Architecture

The model architecture used in this pipeline is Facebook's pre-trained SAM (Segment Anything Model), specifically the `sam-vit-base` variant. The SAM model is a vision transformer-based architecture.

- **Vision Encoder**: This part of the model is frozen during training (no gradient updates).
- **Prompt Encoder**: Used for encoding the bounding box prompts derived from the ground truth masks.
- **Mask Decoder**: The only part of the model being fine-tuned in this pipeline. It is responsible for predicting the segmentation masks.

## Training Process

The training process includes the following steps:

1. **Data Preparation**: Images and masks are processed into 256x256 patches. Only non-empty patches are used for training.
2. **Bounding Box Extraction**: For each mask, a bounding box is calculated around the non-zero pixels (the segmented area).
3. **DataLoader**: A PyTorch DataLoader is used to batch and shuffle the dataset.
4. **Model Training**: The SAM model is trained by backpropagating the loss through the mask decoder's parameters.
5. **Optimizer**: The Adam optimizer is used with a learning rate of `1e-5`.

### Sample Training Loop

```python
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in train_dataloader:
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)
        
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}, Mean loss: {mean(epoch_losses)}')
```

## Saving the Model

The trained model's state dictionary is saved after training. Modify the `model_save_path` variable to specify where the `.pth` file will be stored.

```python
torch.save(model.state_dict(), model_save_path)
```

## Loss Functions

This script supports three different loss functions, which can be chosen by setting the `loss_fn_type` hyperparameter:

1. **DiceCELoss** (default): A combination of Dice and Cross Entropy loss.
2. **FocalLoss**: Focal loss for imbalanced datasets.
3. **DiceFocalLoss**: A combination of Dice and Focal loss.

## Visualization

The script includes a basic visualization tool using `matplotlib` to display an image and its corresponding mask:

```python
import matplotlib.pyplot as plt
img_num = random.randint(0, len(dataset) - 1)
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

fig, axes = plt.subplots(1, 2)
axes[0].imshow(np.array(example_image))
axes[1].imshow(example_mask, cmap='gray')
plt.show()
```

This helps you verify that the images and masks are being correctly processed before training.

## References

- [Facebook SAM Model](https://github.com/facebookresearch/segment-anything)
- [MONAI Loss Functions](https://monai.io)
