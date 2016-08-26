import os
import sys
import glob
import random
import copy
import numpy as np

from APPIL_DNN.config import Config
from APPIL_DNN.data_helper import DataHelper
from APPIL_DNN.path_helper import PathHelper


class ImagePathHelper:


    @staticmethod
    def get_image_list(input_path, dist=None, count=None, exclude=None):
        input_files = PathHelper.glob(os.path.join(input_path, "*.nrrd"))
        random.shuffle(input_files)

        if exclude is None:
            exclude = []

        for x in exclude:
            if x in input_files:
                input_files.remove(x)

        if dist is None:
            if count is None:
                return input_files
            else:
                return input_files[0:count]

        num_examples, num_classes, labels = DataHelper.get_labels()

        if dist is not None:
            if sum(dist) != 1.0:
                max_dist = sum(dist)
                for idx, val in enumerate(dist):
                    dist[idx] /= max_dist

            if count is None:
                count = len(input_files)
                # raise ValueError("You have to specify a total count of samples")

        count_per_class = np.int16((np.array(dist) * count))

        selected_files = []
        for idx, val in enumerate(input_files):
            record_id = ((os.path.splitext(os.path.basename(val))[0]).split('_'))[0]
            label = labels[record_id]
            if count_per_class[label] != 0:
                selected_files.append(val)
                count_per_class[label] -= 1

            if sum(count_per_class) == 0:
                return selected_files

        raise StopIteration("Required number of file could not be fulfilled from input directory")


    @staticmethod
    def get_next_image_path(input_path, dist=None, count=0, exclude=None):
        input_files = PathHelper.glob(os.path.join(input_path, "*.nrrd"))
        random.shuffle(input_files)

        num_examples, num_classes, labels = DataHelper.get_labels()

        if dist is not None:
            if sum(dist) != 1.0:
                max_dist = sum(dist)
                for idx, val in enumerate(dist):
                    dist[idx] /= max_dist

            if count <= 0:
                raise ValueError("You have to specify a total count of samples")

        if exclude is None:
            exclude = []

        for x in exclude:
            if x in input_files:
                input_files.remove(x)

        # No specific distribution? Just grab files randomly
        if dist is None:
            while len(input_files):
                selected_image = random.choice(input_files)
                input_files.remove(selected_image)
                record_id = ((os.path.splitext(os.path.basename(selected_image))[0]).split('_'))[0]
                label = labels[record_id]
                yield selected_image, label
            return

        # build class map
        class_map = list(range(len(input_files)))
        if dist is not None:
            class_map = [[] for i in range(num_classes)]
            for i in range(len(input_files)):
                f = input_files[i]
                record_id = ((os.path.splitext(os.path.basename(f))[0]).split('_'))[0]
                label = labels[record_id]
                class_map[label].append((f, label))

        original_map = copy.deepcopy(class_map)

        count_yielded = 0
        while count_yielded < count:
            selected_class = np.random.choice(num_classes, 1, p=dist)[0]

            if len(class_map[selected_class]) == 0:
                class_map[selected_class] = copy.deepcopy(original_map[selected_class])

            selected = random.choice(class_map[selected_class])
            class_map[selected_class].remove(selected)

            count_yielded += 1
            yield selected