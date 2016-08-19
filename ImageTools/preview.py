#!/usr/bin/env python3

# Example use :
#       python preview.py ..\Data\train\discrete\6\ --output output.png --size 10x10
import numpy as np
import SimpleITK as sitk
import os
import glob
import random
import argparse
import tempfile
from PIL import Image

temp_filename = os.path.abspath(tempfile.gettempdir() + '/' + 'preview_temp.jpg')

parser = argparse.ArgumentParser(description='Preview images from a given directory.')
parser.add_argument('input', type=str, help='Path to input directory')
parser.add_argument('--output', type=str, default=temp_filename, help='[Optional] Path to output image')
parser.add_argument('--size', type=str, default="5x5", help='[Optional] Size of the grid')
args = parser.parse_args()

input_path = args.input
output_path = args.output
grid_size = args.size.split('x')

assert len(grid_size) == 2
grid_size = (int(grid_size[0]), int(grid_size[1]))

input_files = glob.glob(input_path + '/' + "*.nrrd")
random.shuffle(input_files)

images = []
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        counter = i * grid_size[0] + j
        image = sitk.ReadImage(input_files[counter], sitk.sitkFloat32)
        image_size = image.GetSize()
        selected_slice = int(image_size[2] / 2)
        nd_image = sitk.GetArrayFromImage(image)
        slc = nd_image[selected_slice, :, :]
        images.append(sitk.GetImageFromArray(slc))

tiled_image = sitk.Tile(images, grid_size, 128)
#tiled_image = sitk.Cast(sitk.RescaleIntensity(tiled_image, 0, 2**16), sitk.sitkUInt32)
tiled_image = sitk.Cast(sitk.RescaleIntensity(tiled_image, 0, 2**8), sitk.sitkUInt8)

sitk.WriteImage(tiled_image, output_path)

pil_image = Image.open(output_path)
pil_image.show()
