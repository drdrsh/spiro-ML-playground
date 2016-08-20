import numpy as np
import os, sys, subprocess, time, glob, csv, json

class Config:

    instance = None

	class __Config:
		def __init__(self, filename=None):
			try:
				if filename is None:
					filename = './config.json'
				with open(filename) as data_file:
					self.config = json.load(data_file)
			except:
				pass

		def get(self, key):
			return self.config[key]


	def load(filename):
		config = Config.get_instance()
		with open(filename) as data_file:
			config.config = json.load(data_file)
		return config

    @staticmethod
	def get_instance():
		if Config.instance is None:
			Config.instance = Config.__Config()
		return Config.instance

	def get(k):
		config = Config.get_instance()
		return config.get(k)