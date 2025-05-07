import os
import numpy as np
from PIL import Image
from transformers.image_utils import load_image
from segment import ObjectSegmenter  
import torch

def batch_segment_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    segmenter = ObjectSegmenter(
        detector_id="IDEA-Research/grounding-dino-tiny",
        segmenter_id="facebook/sam-vit-base",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    segmenter.to(segmenter.device)

    keywords = [
        "warehouse floor", "factory floor", "concrete floor",
        "gray floor", "cement floor", "epoxy floor", "ground", "floor"
    ]

    success_count = 0
    fail_count = 0
    total_images = 0

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        total_images += 1
        image_path = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")

        try:
            image = load_image(image_path)
            matched = False
            for keyword in keywords:
                mask = segmenter.segment(image=image, keyword=keyword)
                if np.array(mask).sum() > 0:
                    print(f"✓ Success with keyword: {keyword}")
                    matched = True
                    success_count += 1
                    break

            if not matched:
                print("✗ No keyword matched — saving blank mask.")
                mask = Image.new("L", image.size, 0)
                fail_count += 1

            mask_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_mask.png")
            mask.save(mask_path)
            print(f"✓ Saved mask to {mask_path}")

        except Exception as e:
            print(f"✗ Failed on {filename}: {e}")
            fail_count += 1

    print("\nBatch segmentation complete.")
    print(f"Successful segmentations: {success_count}")
    print(f"Failed or no-mask results: {fail_count}")
    print(f"Total images processed: {total_images}")

if __name__ == "__main__":
    input_folder = "/home/abhi/peer/Pallets"
    output_folder = "/home/abhi/peer/floor_masks"
    batch_segment_images(input_folder, output_folder)
