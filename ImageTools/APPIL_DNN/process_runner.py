import csv, sys, os, subprocess, time, logging, datetime

from APPIL_DNN.config import Config


class ProcessRunner:

	def __init__(self, format, max_process=None):

		self.max_process = Config.get('max_process')
		if max_process is not None:
			self.max_process = max_process
		self.processes = []
		self.format = format
		self.files_done = 0
		self.files_total= 0
		self.queue = []
		self.logger = logging.getLogger('process_runner')
		handler = logging.FileHandler('process_runner.log')
		handler.setLevel(logging.INFO)
		self.logger.addHandler(handler)

	def enqueue(self, file_count, params):
		self.files_total += file_count
		self.queue.append( (file_count, params) )


	def run(self):

		self.update_cli()

		str_now = datetime.datetime.now().strftime("[%d-%m-%Y %H:%M:%S]")
		self.logger.info('{0} Started processing {1} files'.format(str_now, self.files_total))
		for file_count, params in self.queue:
			if len(self.processes) < self.max_process:
				with open(os.devnull, 'w') as fp:
					p = subprocess.Popen(params, stdout=fp, stderr=fp)
					p.count = file_count
					p.cmd = " ".join(params)
					p.started =  datetime.datetime.now()
					p.ended = None
					self.processes.append(p)
			else:
				self.wait_for_empty_spot()

		self.wait_for_empty_queue()

	def update_cli(self):
		total_pct = 100 if self.files_total == 0 else  (self.files_done / self.files_total) * 100

		sys.stdout.write(
			(self.format).format(
				self.files_done,
				self.files_total,
				total_pct
			)
		)
		sys.stdout.flush()


	def wait_for_empty_spot(self):
		while self.processes:
			for proc in self.processes:
				retcode = proc.poll()
				if retcode is not None: # Process finished.
					proc.ended = datetime.datetime.now()
					out, err = proc.communicate()
					started = proc.started.strftime("[%d-%m-%Y %H:%M:%S]")
					ended = proc.ended.strftime("[%d-%m-%Y %H:%M:%S]")

					if retcode == 0:
						self.logger.debug(
							'Command {0} called on {1} ended in {2}\n{3}'
								.format(proc.cmd, started, ended, out)
						)
					else :
						self.logger.error(
							'<ERROR> Command {0} called on {1} ended in {2} caused an error\n{3}'
								.format(proc.cmd, started, ended, err)
						)
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