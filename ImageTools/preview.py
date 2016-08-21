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

from APPIL_DNN.cli import CLI

temp_filename = os.path.abspath(tempfile.gettempdir() + '/' + 'preview_temp.jpg')

parser = argparse.ArgumentParser(description='Preview images from a given directory.')
parser.add_argument('input', type=str, help='Path to input directory')
parser.add_argument('--output', type=str, default=temp_filename, help='[Optional] Path to output image')
parser.add_argument('--size', type=str, default="5x5", help='[Optional] Size of the grid')
parser.add_argument('--slice', type=float, default=0.5, help='[Optional] Slice index as 0.0 to 1.0, -1 for random')
parser.add_argument('-n', action='store_true', default=False, help='[Optional] enable numpy mode (reads .npy files')

args = parser.parse_args()

input_path = args.input
output_path = args.output
grid_size = args.size.split('x')
is_numpy = args.n
slice_pct = args.slice

assert len(grid_size) == 2
grid_size = (int(grid_size[0]), int(grid_size[1]))

def read_npy(input, slice_selection=.5):

    input_files = glob.glob(input + "/data_*")

    def get_next_image(file_list) :
        current_index = -1
        current_batch = None
        while True:
            if current_index == -1  or current_index >= current_batch.shape[0]:
                if len(file_list) == 0:
                    return
                current_index = 0
                f = random.choice(file_list)
                file_list.remove(f)
                current_batch = np.load(f)

            yield current_batch[current_index]
            current_index += 1

    images = []
    image_gen = get_next_image(input_files)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            nd_image = next(image_gen)
            slice_pct = slice_selection
            if slice_selection < 0:
                slice_pct = random.random()
            selected_slice = int(nd_image.shape[0] * slice_pct)
            slc = nd_image[selected_slice, :, :]
            images.append(sitk.GetImageFromArray(slc))
            if len(images) == grid_size[0] * grid_size[1]:
                return images



def read_nrrd(input, slice_selection=.5):

    input_files = glob.glob(input + "/*.nrrd")
    random.shuffle(input_files)

    images = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            counter = i * grid_size[0] + j
            is_error = True
            while is_error:
                try:
                    image = sitk.ReadImage(input_files[counter], sitk.sitkFloat32)
                    is_error = False
                except RuntimeError:
                    counter += 1
                    if counter >= len(input_files):
                        CLI.exit_error("Not enough valid images to create a {0}x{1} grid".format(*grid_size))
                    is_error = True

            image_size = image.GetSize()
            slice_pct = slice_selection
            if slice_selection < 0:
                slice_pct = random.random()
            selected_slice = int(image_size[2] * slice_pct)
            nd_image = sitk.GetArrayFromImage(image)
            slc = nd_image[selected_slice, :, :]
            images.append(sitk.GetImageFromArray(slc))

    return images



if is_numpy:
    images = read_npy(input_path, slice_pct)
else :
    images = read_nrrd(input_path, slice_pct)

tiled_image = sitk.Tile(images, grid_size, 128)
# tiled_image = sitk.Cast(sitk.RescaleIntensity(tiled_image, 0, 2**16), sitk.sitkUInt32)
tiled_image = sitk.Cast(sitk.RescaleIntensity(tiled_image, 0, 2**8), sitk.sitkUInt8)

sitk.WriteImage(tiled_image, output_path)

pil_image = Image.open(output_path)
pil_image.show()


