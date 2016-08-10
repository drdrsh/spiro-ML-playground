import SimpleITK as sitk
import numpy as np
import csv, os, glob, sys, random

total_files = 0
files_done = 0
batches_written = 0

def update_cli():
	global total_files, files_done
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


shrink_factor = 10
root_path     = "/home/mostafa/SummerProject/Data/"
labels_path   = root_path + "labels.csv"
images_path   = root_path + "/augmented/" + str(shrink_factor) + "_trial/"
np_path       = root_path + "/np/" + str(shrink_factor) + "_trial/"


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
random.shuffle(images)

total_files = len(images)

for image_path in images:

	update_cli()

	# print("Processing subject {0})".format(record_id))
	record_id = ((os.path.splitext(os.path.basename(image_path))[0]).split('_'))[0]
	labels = labels_table[record_id]

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
		write_batch(X, Y, batch_number, np_path)
		X = []
		Y = []
		cum_batch_size = 0
		batch_number += 1
		batches_written += 1

	cum_batch_size += arr.nbytes

	cls = [0, 0]
	cls[labels['emphysema']] = 1
	X.append(arr)
	Y.append(cls)

	files_done += 1

# print("Writing last batch")
write_batch(X, Y, batch_number, np_path)
batches_written += 1
update_cli()
print("Done!")
