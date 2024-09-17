import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO

def create_masks(coco_json_path, images_dir, output_dir):
    # Load COCO annotations
    coco = COCO(coco_json_path)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image IDs from the COCO dataset
    img_ids = coco.getImgIds()
    
    # Iterate through each image ID
    for img_id in img_ids:
        img_info = coco.imgs[img_id]
        img_filename = img_info['file_name']
        img_path = os.path.join(images_dir, img_filename)

        # Load the image to get dimensions
        if not os.path.exists(img_path):
            print(f"Image {img_filename} not found in {images_dir}. Skipping...")
            continue
        
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # Get annotations for the current image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # Create a mask for each class
        masks_by_class = {}
        
        for ann in annotations:
            class_id = ann['category_id']
            class_name = coco.cats[class_id]['name']

            # Create a mask for the current annotation
            mask = coco.annToMask(ann)

            # Initialize the class directory if it doesn't exist
            class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Combine mask with existing masks for this class
            if class_id not in masks_by_class:
                masks_by_class[class_id] = np.zeros((height, width), dtype=np.uint8)
            
            masks_by_class[class_id] = np.maximum(masks_by_class[class_id], mask)

        # Save masks for each class
        for class_id, combined_mask in masks_by_class.items():
            class_name = coco.cats[class_id]['name']
            mask_filename = f"{os.path.splitext(img_filename)[0]}_{class_name}.png"
            mask_path = os.path.join(output_dir, class_name, mask_filename)

            # Save the mask as a binary image (0-255)
            cv2.imwrite(mask_path, combined_mask * 255)

    print(f"Mask generation completed. Masks saved to: {output_dir}")

# Hard-coded paths for demonstration purposes
coco_json_path = r'E:\Random Python Scripts\Tata HaxS\Car dentss.v1i.coco-segmentation\test\_annotations.coco.json'  # Update this path
images_dir = r'E:\Random Python Scripts\Tata HaxS\Car dentss.v1i.coco-segmentation\test'                # Update this path
output_dir = r'E:\Random Python Scripts\Tata HaxS\Car dentss.v1i.coco-segmentation\test\masks'          # Update this path

create_masks(coco_json_path, images_dir, output_dir)