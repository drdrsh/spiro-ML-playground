import SimpleITK as sitk
import numpy as np
import csv, os


def pad_image(image, target_dim):
	arr = sitk.GetArrayFromImage(image)

	padding_map = []
	for i in range(len(target_dim)):
		padding_map.append( [0, max(0, target_dim[i] - arr.shape[i])] )

	x = np.pad(arr, padding_map, mode='constant',constant_values=0)
	return np.int16(x)


root_path     = "/home/mostafa/SummerProject/Data/"
labels_path   = root_path + "labels.csv"
series_path   = root_path + "series/"
np_path       = root_path + "np/"

shrink_factors= range(2, 4)
ref_dim       = (612, 512, 512)
batch_max_size= 3.5 * 1024 * 1024 * 1024 # 3.5 Gigabytes
batch_max_size= 200 * 1024 * 1024 		 # 900 Megabytes

cum_batch_size = 0
for shrink_factor in shrink_factors:

	batch_number = 0
	entries = []
	scale_factor  = 1/shrink_factor
	out_dim = (int(ref_dim[0] * scale_factor), int(ref_dim[1] * scale_factor), int(ref_dim[2] * scale_factor))

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

			subject_path = series_path + record_id
			if not os.path.isdir(subject_path):
				print("Missing subject {0}".format(subject_path))
				continue

			print("Processing subject {0} at shrink factor {1} ({2}x{3}x{4})".format(record_id, shrink_factor, out_dim[0], out_dim[1], out_dim[2]))
			reader = sitk.ImageSeriesReader()
			reader.SetFileNames(reader.GetGDCMSeriesFileNames(subject_path))
			image = reader.Execute()

			image = sitk.Shrink(image, [shrink_factor, shrink_factor, shrink_factor])

			arr = pad_image(image, out_dim)
			print("Array size = {0} bytes".format(arr.nbytes))

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
			print(cum_batch_size)
			# For simplicity just use emphysema status as the class variable
			entries.append([arr, emphysema])

		arr = np.array(entries)
		filename = "data_{0}_{1}".format(shrink_factor, batch_number)
		print("Writing file " + filename)
		np.savez(np_path + filename, arr)

