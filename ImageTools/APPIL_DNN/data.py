import csv, os
from APPIL_DNN.config import Config




def get_labels():

	root = Config.get('data_root')
	labels_file = os.path.abspath('/'.join([root, 'labels.csv']))
	num_examples = 0
	num_classes = 0
	result = {}
	with open(labels_file, 'r') as csvfile:
		csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
		for row in csv_reader:
			record_id = row['record_id']
			num_examples += 1

			emphysema = 0 if row['emphysema'] == '' else int(row['emphysema'])
			bronchiectasis = 0 if row['bronchiectasis1'] == '' else int(row['bronchiectasis1'])

			cls = -1

			if emphysema == 0 and bronchiectasis == 0:
				cls = 0
			if emphysema == 1 and bronchiectasis == 0:
				cls = 1
			if emphysema == 0 and bronchiectasis == 1:
				cls = 2
			if emphysema == 1 and bronchiectasis == 1:
				cls = 3

			# This will set up a 4 class dataset
			# result[record_id] = cls
			# num_classes = 4

			# This will set up a 2 class dataset
			result[record_id] = emphysema
			num_classes = 2

		return num_examples, num_classes, result
