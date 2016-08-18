import csv, os
import numpy as np
import threading
from APPIL_DNN.config import Config

class RunningStat:

	# FIXME: See why variance sometimes ends up being negative
	def __init__(self):
		self.lock = threading.Lock()
		self.K = 0
		self.n = 0
		self.ex = 0
		self.ex2 = 0


	def add_batch(self, x):

		self.lock.acquire()

		flat_x = x.flatten()

		if self.n == 0:
			self.K = flat_x[0]

		x_k  = flat_x - self.K
		x2_k = np.power(flat_x - self.K, 2)

		x_k_sum = np.sum(x_k)
		x2_k_sum= np.sum(x2_k)

		self.n   += flat_x.shape[0]
		self.ex  += x_k_sum
		self.ex2 += x2_k_sum

		self.lock.release()


	def add_variable(self, x):
		if self.n == 0:
			self.K = x
		self.n = self.n + 1
		self.ex += x - self.K
		self.ex2 += (x - self.K) * (x - self.K)

	def remove_variable(self, x):
		self.n = self.n - 1
		self.ex -= (x - self.K)
		self.ex2 -= (x - self.K) * (x - self.K)

	def get_meanvalue(self):
		return self.K + self.ex / self.n

	def get_variance(self):
		return (self.ex2 - (self.ex*self.ex)/self.n) / (self.n-1)


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
