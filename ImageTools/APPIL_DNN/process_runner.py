import datetime
import logging
import os
import subprocess
import sys
import time
import threading
from queue import *

from APPIL_DNN.config import Config
from APPIL_DNN.basic_stdout import BasicStdout


class ProcessRunner:

    def __init__(self, fmt, max_process=None):

        self.max_process = Config.get('max_process')
        if max_process is not None:
            self.max_process = max_process
        self.threads = []
        self.q = Queue()
        self.std_printer = BasicStdout.get_instance()
        self.std_printer.set_format(fmt)
        self.std_printer.set_variables({
            'total_files': 0,
            'files_done': 0,
            'files_done_pct': 0
        })

        formatter = logging.Formatter("[%(asctime)s]  %(name)-12s %(levelname)s: %(message)s", style="{")

        self.logger = logging.getLogger('process_runner')
        file_handler = logging.FileHandler("process_runner.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        self.exit_flag = False

    def enqueue(self, file_count, params):
        total_files = self.std_printer.get_variable('total_files')
        self.std_printer.set_variable('total_files', total_files + file_count)
        self.q.put((file_count, params))

    def perform_task(self):

        while not self.exit_flag:
            data = self.q.get()
            if data is None:
                self.q.task_done()
                break
            file_count, params = data

            started = datetime.datetime.now()
            self.logger.debug("Process {cmd} started at {time}".format(**{
                'cmd': " ".join(params),
                'time': started.strftime('[%d-%m-%Y %H:%M:%S]')
            }))
            result = subprocess.run(
                params,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            ended = datetime.datetime.now()
            elapsed = ended - started
            self.logger.debug("Process {cmd} ended at {time} and ran in {elapsed} seconds : \n {out}".format(**{
                'cmd': " ".join(params),
                'out': result.stdout,
                'time': ended.strftime('[%d-%m-%Y %H:%M:%S]'),
                'elapsed': elapsed.seconds
            }))

            if len(result.stderr):
                self.logger.error("Process {cmd} returned an error : \n {err}".format(**{
                    'cmd': " ".join(params),
                    'err': result.stderr
                }))

            self.logger.handlers[0].flush()
            files_done = self.std_printer.get_variable('files_done') + file_count
            total_files = self.std_printer.get_variable('total_files')

            self.std_printer.set_variable('files_done', files_done)
            self.std_printer.set_variable('files_done_pct', (files_done / total_files) * 100)

            self.q.task_done()

    def cancel(self):
        self.exit_flag = True

    def start(self):

        for c in range(self.max_process):
            t = threading.Thread(target=self.perform_task)
            t.daemon = True
            t.start()
            self.threads.append(t)
            self.q.put(None)

        try:
            while True:
                time.sleep(100)
                # self.q.

        except (KeyboardInterrupt, SystemExit):
            print('\nQuitting, please wait....\n')
            self.cancel()
            # self.q.join()
            for t in self.threads:
                t.join()
