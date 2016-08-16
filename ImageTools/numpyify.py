#!/usr/bin/env python3

import SimpleITK as sitk
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
	np.save(np_path + '/' + filename, np.array(X))

	filename = "labels_{0}".format(batch_number)
	# print("Writing file " + filename)
	np.save(np_path + '/' + filename, np.array(Y))



def pad_image(image, target_dim):

	size = image.GetSize()

	trim_start_list = [0, 0, 0]
	trim_end_list = [0, 0, 0]

	for i in range(len(size)):

		to_trim = 0
		if size[i] > target_dim[i]:
			to_trim = size[i] - target_dim[i]
		if  int( (to_trim * 100) / target_dim[i]) >= 50:
			# Difference between image size and target size is different by more than 50%
			# Trimming will likely destroy image data, discard the image altogether
			return None

		trim_start = int(to_trim / 2)
		trim_end   = trim_start + (to_trim % 2)
		trim_start_list[i] = trim_start
		trim_end_list[i] = trim_end

	image = sitk.Crop(image, trim_start_list, trim_end_list)

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


def process_path(input_path, output_path, labels_table, num_classes, out_dim) :

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

	# print(np.array(X).shape)
	# print("Writing last batch")
	write_batch(X, Y, batch_number, output_path)
	batches_written += 1
	update_cli(total_files, files_done, batches_written)
	print("\nDone!\n")


if len(sys.argv) > 1:
	Config.load(sys.argv[1])

current_mode = 'train'
if len(sys.argv) > 2:
	current_mode = argv[2]


ref_dim = Config.get('reference_dimensions')
segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
batch_max_size = Config.get('batch_max_size')

output_dimensions = None
try:
	# This will override the calculate dimensions, note that dimensions will be reduced by trimming from both sides
	output_dimensions = tuple(Config.get('output_dimensions'))
except KeyError:
	pass

num_examples, num_classes, labels_table = APPIL_DNN.data.get_labels()

scale_factor  = 1/active_shrink_factor

out_dim = (int(ref_dim[0] * scale_factor), int(ref_dim[1] * scale_factor), int(ref_dim[2] * scale_factor))
if output_dimensions is not None:
	out_dim = output_dimensions

print('\nOutput image size is ' + str(out_dim) + '\n')

runs = {
	'train': ['raw_augmented', 'segmented_augmented'],
	'test' : ['raw', 'segmented']
}

for i in runs:

	print('\nProcessing {0} data\n'.format(i))

	input_type = i
	mode = 1 if segment_enabled else 0
	input_subtype = runs[i][mode]
	input_path  = CLI.get_path(i , input_subtype,  active_shrink_factor)
	output_path = CLI.get_path(i, input_subtype + '_np', active_shrink_factor)
	try:
		os.makedirs(output_path)
	except FileExistsError:
		pass
	process_path(input_path, output_path, labels_table, num_classes, out_dim)