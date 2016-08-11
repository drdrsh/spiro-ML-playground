import numpy as np
import os, sys, subprocess, time, glob, csv


class Config:

	@staticmethod
	def get(key):
		config = {
			'bin_root'	   : '/home/mostafa/SummerProject/ImageTools/bin/',
			'data_root'    : '/home/mostafa/SummerProject/Data/',
			'test_set_size': 24,
			'max_process'  : 8,
			'segment_enabled': True,
			'min_augment_count': 50,
			'max_augment_count': 100,
			'active_shrink_factor': 5,
			'reference_dimensions': (630, 512, 512),
			'batch_max_size': 200 * 1024 * 1024
		}
		return config[key]