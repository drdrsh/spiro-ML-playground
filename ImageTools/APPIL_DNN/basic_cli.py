import threading
import sys

class BasicCLI:
    def __init__(self):
        self.lock = threading.Lock()
        self.format = ""
        self.variables = {}


    def set_format(self, format):
        self.format = format

    def get_variable(self, key):
        return self.variables[key]

    def set_variable(self, key, value):
        self.lock.acquire()
        self.variables[key] = value
        self.lock.release()
        self.update()

    def set_variables(self, variables):
        self.variables = variables

    def add_line(self, text):
        self.lock.acquire()
        print("\n" + text + "\n".format(**self.variables))
        self.lock.release()

    def update(self, variables={}):
        full_dict = {**variables, **self.variables}
        self.lock.acquire()
        sys.stdout.write(self.format.format(**full_dict))
        sys.stdout.flush()
        self.lock.release()
