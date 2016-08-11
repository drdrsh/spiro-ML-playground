#!/usr/bin/env python3

# import SimpleITK as sitk
import numpy as np
import csv, os, glob, sys, random

import APPIL_DNN.data
from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner


def update_cli(total_files, files_done, batches_written ):
	sys.stdout.write(
		'Converting image to numpy array, {0} out of {1} files processed ({3} batches written) ({2:.2f}%)\r'
			.format(
				files_done,
				total_files,
				(files_done/total_files) * 100 ,
				batches_written
			)
	)
	sys.stdout.flush()


def write_batch(X, Y, batch_number, np_path):
	filename = "data_{0}".format(batch_number)
	# print("Writing file " + filename)
	np.save(np_path + filename, np.array(X))

	filename = "labels_{0}".format(batch_number)
	# print("Writing file " + filename)
	np.save(np_path + filename, np.array(Y))



def pad_image(image, target_dim):

	size = image.GetSize()
	if size[2] > target_dim[0]:

		to_trim = size[2] - target_dim[0]
		if  int( (to_trim * 100) / target_dim[0]) >= 25:
			# Difference between image size and target size is different by more than 25%
			# Trimming will likely destroy image data, discard the image altogether
			return None

		trim_start = int(to_trim / 2)
		trim_end   = trim_start + (to_trim % 2)

		image = sitk.Crop(image, [0, 0, trim_start], [0, 0, trim_end] )

	arr = sitk.GetArrayFromImage(image)

	padding_map = []
	for i in range(len(target_dim)):
		padding_map.append( [0, max(0, target_dim[i] - arr.shape[i])] )

	# TODO: Resample images that are larger after padding (images with z-spacing < 1.0)

	x = np.pad(arr, padding_map, mode='constant',constant_values=0)

	# Zero center and normalize data
	x = np.float32(x)
	x -= np.mean(x)
	x /= np.std(x)

	return np.int16(x)


current_mode = 'train'

ref_dim = Config.get('reference_dimensions')
segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
batch_max_size = Config.get('batch_max_size')


input_subtype  = 'segmented_augmented' if segment_enabled  else 'augmented'
input_path  = CLI.get_path(current_mode , input_subtype,  active_shrink_factor)

output_path = CLI.get_path(current_mode , 'np', active_shrink_factor)

try:
	os.mkdir(output_path)
except:
	pass

num_examples, num_classes, labels_table = APPIL_DNN.data.get_labels()

scale_factor  = 1/active_shrink_factor
out_dim = (int(ref_dim[0] * scale_factor), int(ref_dim[1] * scale_factor), int(ref_dim[2] * scale_factor))


print('Output image size is ' + str(out_dim))
images = glob.glob(input_path + '/' + "*.nrrd")
random.shuffle(images)

files_done = 0
batches_written = 0
total_files = len(images)

cum_batch_size = 0
batch_number = 0
cls = list(0 for i in range(num_classes))
X = []
Y = []

assert len(images) != 0

for image_path in images:

	update_cli(total_files, files_done, batches_written)

	# print("Processing subject {0})".format(record_id))
	record_id = ((os.path.splitext(os.path.basename(image_path))[0]).split('_'))[0]
	label = labels_table[record_id]
	cls[label] = 1

	try:
		image = sitk.ReadImage(image_path)
	except(RuntimeError):
		#print("Invalid image {0}".format(image_path))
		files_done += 1
		continue

	arr = pad_image(image, out_dim)
	if arr is None:
		#print("Discarding subject {0}\n".format(record_id))
		files_done += 1
		continue

	# print("Array size = {0} bytes".format(arr.nbytes))

	if cum_batch_size + arr.nbytes > batch_max_size:

		# print("Batch size reached, starting a new batch")
		write_batch(X, Y, batch_number, output_path)
		X = []
		Y = []
		cum_batch_size = 0
		batch_number += 1
		batches_written += 1

	cum_batch_size += arr.nbytes

	X.append(arr)
	Y.append(cls)

	files_done += 1

# print("Writing last batch")
write_batch(X, Y, batch_number, output_path)
batches_written += 1
update_cli(total_files, files_done, batches_written)
print("Done!")
