import csv, sys, os, subprocess, time
from APPIL_DNN.config import Config


class ProcessRunner:

	def __init__(self, format):

		self.max_process = Config.get('max_process')
		self.processes = []
		self.format = format
		self.files_done = 0
		self.files_total= 0
		self.queue = []

	def enqueue(self, file_count, params):
		self.files_total += file_count
		self.queue.append( (file_count, params) )


	def run(self):

		self.update_cli()
		for file_count, params in self.queue:
			if len(self.processes) < self.max_process:
				with open(os.devnull, 'w') as fp:
					p = subprocess.Popen(params, stdout=fp)
					p.count = file_count
					self.processes.append(p)
			else:
				self.wait_for_empty_spot()

		self.wait_for_empty_queue()

	def update_cli(self):
		sys.stdout.write(
			(self.format).format(
				self.files_done,
				self.files_total,
				(self.files_done / self.files_total) * 100
			)
		)
		sys.stdout.flush()


	def wait_for_empty_spot(self):
		while self.processes:
			for proc in self.processes:
				retcode = proc.poll()
				if retcode is not None: # Process finished.
					self.processes.remove(proc)
					self.files_done += proc.count
					self.update_cli()
					break
				else: # No process is done, wait a bit and check again.
					time.sleep(.1)
					continue

	def wait_for_empty_queue(self):
		while self.processes:
			for proc in self.processes:
				retcode = proc.poll()
				if retcode is not None: # Process finished.
					self.processes.remove(proc)
					self.files_done += proc.count
					self.update_cli()
				else: # No process is done, wait a bit and check again.
					time.sleep(.1)