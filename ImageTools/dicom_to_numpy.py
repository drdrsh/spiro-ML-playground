import SimpleITK as sitk
import numpy as np
import csv, os, glob, sys

def write_batch(X, Y, batch_number, np_path):
	filename = "data_{0}".format(batch_number)
	print("Writing file " + filename)
	np.save(np_path + filename, np.array(X))

	filename = "labels_{0}".format(batch_number)
	print("Writing file " + filename)
	np.save(np_path + filename, np.array(Y))

def get_label_dict(labels_file):
	result = {}
	with open(labels_file, 'r') as csvfile:
		csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
		for row in csv_reader:
			record_id = row['record_id']
			result[record_id] = {}
			if row['emphysema'] != '':
				result[record_id]['emphysema'] = int(row['emphysema'])
			else:
				result[record_id]['emphysema'] = 0

			if row['bronchiectasis1'] != '':
				result[record_id]['bronchiectasis'] = int(row['bronchiectasis1'])
			else:
				result[record_id]['bronchiectasis'] = 0
	return result



def pad_image(image, target_dim):

	size = image.GetSize()

	if size[2] > target_dim[0] or size[1] > target_dim[1] or size[0] > target_dim[2]:
		# If the image size is larger than the agreed upon padded size, discard it
		return None

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


shrink_factor = 7
root_path     = "/home/mostafa/SummerProject/Data/"
labels_path   = root_path + "labels.csv"
images_path   = root_path + "/augmented/" + str(shrink_factor) + "/"
np_path       = root_path + "/np/" + str(shrink_factor) + "/"


labels_table = get_label_dict(labels_path)

ref_dim       = (630, 512, 512)
batch_max_size= 3.5 * 1024 * 1024 * 1024 # 3.5 Gigabytes
batch_max_size= 200 * 1024 * 1024 		 # 200 Megabytes



scale_factor  = 1/shrink_factor
out_dim = (int(ref_dim[0] * scale_factor), int(ref_dim[1] * scale_factor), int(ref_dim[2] * scale_factor))

cum_batch_size = 0
batch_number = 0
X = []
Y = []

images = glob.glob(images_path + "*.nrrd")

for image_path in images:

	# print("Processing subject {0})".format(record_id))
	record_id = ((os.path.splitext(os.path.basename(image_path))[0]).split('_'))[0]
	labels = labels_table[record_id]

	image = sitk.ReadImage(image_path)
	arr = pad_image(image, out_dim)
	if arr is None:
		print("Discarding subject {0}".format(record_id))
		continue

	# print("Array size = {0} bytes".format(arr.nbytes))

	if cum_batch_size + arr.nbytes > batch_max_size:

		print("Batch size reached, starting a new batch")
		write_batch(X, Y, batch_number, np_path)
		X = []
		Y = []
		cum_batch_size = 0
		batch_number += 1

	cum_batch_size += arr.nbytes

	cls = [0, 0]
	cls[labels['emphysema']] = 1
	X.append(arr)
	Y.append(cls)

print("Writing last batch")
write_batch(X, Y, batch_number, np_path)
