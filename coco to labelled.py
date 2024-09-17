import json
import numpy as np
import skimage.draw
from PIL import Image
import os
import shutil


def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint16)

    # Counter for the object number
    object_number = 1

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            # Extract segmentation polygon
            for seg in ann['segmentation']:
                # Convert polygons to a binary mask and add it to the main mask
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                mask_np[rr, cc] = object_number
                object_number += 1  # Assigning each object a unique integer value (labeled mask)

    # Save the numpy array as a JPEG using PIL
    mask_image = Image.fromarray(mask_np.astype(np.uint8))  # Convert to uint8 for saving as JPEG
    mask_path = os.path.join(output_folder, image_info['file_name'].replace('.jpg', '_mask.jpg'))
    mask_image.save(mask_path)

    print(f"Saved mask for {image_info['file_name']} to {mask_path}")


def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load COCO JSON annotations
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for img in images:
        # Create the masks
        create_mask(img, annotations, mask_output_folder)
        
        # Copy original images to the specified folder
        original_image_path = os.path.join(original_image_dir, img['file_name'])
    
        new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
        shutil.copy2(original_image_path, new_image_path)
        print(f"Copied original image to {new_image_path}")


if __name__ == '__main__':
    original_image_dir = r'E:\Random Python Scripts\Tata HaxS\Car dentss.v1i.coco-segmentation-copy\train'  
    json_file = r"E:\Random Python Scripts\Tata HaxS\Car dentss.v1i.coco-segmentation-copy\train\_annotations.coco.json"
    mask_output_folder = r'E:\Random Python Scripts\Tata HaxS\Car dentss.v1i.coco-segmentation-copy\train\masks'  
    image_output_folder = r'E:\Random Python Scripts\Tata HaxS\Car dentss.v1i.coco-segmentation-copy\train\imgs2'  
    main(json_file, mask_output_folder, image_output_folder, original_image_dir)
