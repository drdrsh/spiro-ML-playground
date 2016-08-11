import os, glob, sys, random

from APPIL_DNN.config import Config

class CLI:

	def get_path(typ, subtype, shrink_factor):
		root = Config.get('data_root')
		return os.path.abspath('/'.join([root, str(typ), str(subtype), str(shrink_factor)]))