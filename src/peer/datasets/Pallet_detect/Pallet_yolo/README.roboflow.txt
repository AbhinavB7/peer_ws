
Pallet Detection - v3 2025-05-08 3:41am
==============================

This dataset was exported via roboflow.com on May 8, 2025 at 7:45 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1283 images.
Pallet are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random rotation of between -8 and +8 degrees
* Random brigthness adjustment of between -19 and +19 percent
* Random Gaussian blur of between 0 and 1.5 pixels
* Salt and pepper noise was applied to 0.69 percent of pixels


