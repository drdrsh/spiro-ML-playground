import SimpleITK as sitk
import numpy as np
import csv, os


def pad_image(image, target_dim):

	size = image.GetSize()


	if size[2] > target_dim[0] or size[1] > target_dim[1] or size[0] > target_dim[2]:
		# If the image size is larger than the agreed upon padded size, discard it
		return None

	#filter = sitk.ResampleImageFilter()
	#filter.SetOutputOrigin(image.GetOrigin())
	#filter.SetOutputDirection(image.GetDirection())
	#filter.SetOutputSpacing([3.0, 3.0, 2.5])
	#filter.SetInterpolator(sitk.sitkLinear)
	#image = filter.Execute(image)

	arr = sitk.GetArrayFromImage(image)

	padding_map = []
	for i in range(len(target_dim)):
		padding_map.append( [0, max(0, target_dim[i] - arr.shape[i])] )

	# TODO: Resample images that are larger after padding (images with z-spacing < 1.0)

	x = np.pad(arr, padding_map, mode='constant',constant_values=0)


	x = np.float32(x)
	x -= np.mean(x)
	x /= np.std(x)

	return np.int16(x)


shrink_factor = 7
root_path     = "/home/mostafa/SummerProject/Data/"
labels_path   = root_path + "labels.csv"
image_path    = root_path + "/segmented/" + str(shrink_factor) + "/"
np_path       = root_path + "/np/" + str(shrink_factor) + "/"

ref_dim       = (630, 512, 512)
batch_max_size= 3.5 * 1024 * 1024 * 1024 # 3.5 Gigabytes
# batch_max_size= 200 * 1024 * 1024 		 # 900 Megabytes


scale_factor  = 1/shrink_factor
out_dim = (int(ref_dim[0] * scale_factor), int(ref_dim[1] * scale_factor), int(ref_dim[2] * scale_factor))

cum_batch_size = 0
batch_number = 0
X = []
Y = []
with open(labels_path, 'r') as csvfile:
	csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
	for row in csv_reader:
		record_id = row['record_id']
		if row['emphysema'] != '':
			emphysema = int(row['emphysema'])
		else:
			emphysema = 0

		if row['bronchiectasis1'] != '':
			bronchiectasis = int(row['bronchiectasis1'])
		else:
			bronchiectasis = 0

		subject_path = image_path + record_id + ".nrrd"
		if not os.path.isfile(subject_path):
			# print("Missing subject {0}".format(subject_path))
			continue

		# print("Processing subject {0})".format(record_id))

		image = sitk.ReadImage(subject_path)
		arr = pad_image(image, out_dim)
		if arr is None:
			print("Discarding subject {0}".format(subject_path))
			continue

		# print("Array size = {0} bytes".format(arr.nbytes))

		if cum_batch_size + arr.nbytes > batch_max_size:

			print("Batch size reached, starting a new batch")
			array_to_store = np.array(entries)
			filename = "data_{0}_{1}".format(shrink_factor, batch_number)
			print("Writing file " + filename)
			np.savez(np_path + filename, array_to_store)
			del array_to_store
			del entries
			entries = []
			cum_batch_size = 0
			batch_number += 1

		cum_batch_size += arr.nbytes

		# Flatten array
		# arr.shape = (-1)

		cls = [0, 0]
		cls[emphysema] = 1
		X.append(arr)
		Y.append(cls)

	filename = "images_{0}".format(batch_number)
	print("Writing file " + filename)
	np.save(np_path + filename, np.array(X))

	filename = "labels_{0}".format(batch_number)
	print("Writing file " + filename)
	np.save(np_path + filename, np.array(Y))

