import os
import sys

from APPIL_DNN.config import Config


class CLI:

    @staticmethod
    def get_path(typ, subtype, shrink_factor, prefix=""):
        root = Config.get('data_root')
        return os.path.abspath('/'.join([prefix, root, str(typ), str(subtype), str(shrink_factor)]))

    @staticmethod
    def exit_error(message):
        sys.stderr.write("\n Error: {0}".format(message))
        sys.stderr.flush()
        sys.exit(1)
