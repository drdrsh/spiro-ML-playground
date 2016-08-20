#!/usr/bin/env python3

import SimpleITK as sitk
import numpy as np
import glob
import sys
import random
import math
from queue import *

from APPIL_DNN.data import *
from APPIL_DNN.basic_cli import BasicCLI
from APPIL_DNN.cli import CLI

std_printer = BasicCLI()


def write_batch(X, Y, batch_number, np_path):
    filename = "data_{0}".format(batch_number)
    # print("Writing file " + filename)
    np.save(np_path + '/' + filename, np.array(X))

    filename = "labels_{0}".format(batch_number)
    # print("Writing file " + filename)
    np.save(np_path + '/' + filename, np.array(Y))


def read_image(image_path, target_dim):
    try:
        image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    except RuntimeError:
        return None

    return pad_image(image, target_dim)


def pad_image(image, target_dim):
    size = image.GetSize()

    trim_start_list = [0, 0, 0]
    trim_end_list = [0, 0, 0]

    # This will trim down a image to a target size by trimming two sides in the requested dimension
    for i in range(len(size)):
        to_trim = 0
        if size[i] > target_dim[i]:
            to_trim = size[i] - target_dim[i]
        if int((to_trim * 100) / target_dim[i]) >= 50:
            # Difference between image size and target size is different by more than 50%
            # Trimming will likely destroy image data, discard the image altogether
            return None

        trim_start = int(to_trim / 2)
        trim_end = trim_start + (to_trim % 2)
        trim_start_list[i] = trim_start
        trim_end_list[i] = trim_end

    image = sitk.Crop(image, trim_start_list, trim_end_list)

    arr = sitk.GetArrayFromImage(image)

    # This will zero-pad an image to meet a specific size
    padding_map = []
    for i in range(len(target_dim)):
        padding_map.append([0, max(0, target_dim[i] - arr.shape[i])])

    x = np.pad(arr, padding_map, mode='constant', constant_values=0)

    return np.float32(x)


def stat_worker(q, rs, target_dim, std_printer):
    while True:
        image_path = q.get()
        if image_path is None:
            q.task_done()
            break

        arr = read_image(image_path, target_dim)
        if arr is not None:
            rs.add_batch(arr)

        files_done = std_printer.get_variable('files_done') + 1
        total_files = std_printer.get_variable('total_files')

        std_printer.set_variable('files_done', files_done)
        std_printer.set_variable('files_done_pct', (files_done / total_files) * 100)

        q.task_done()


def get_data_stats(input_path, out_dim):
    global std_printer, Config
    rs = RunningStat()

    images = glob.glob(input_path + "/*.nrrd")
    max_threads = Config.get('max_process')

    total_files = len(images)

    std_printer.set_format(
        'Gathering image statistics {files_done} out of {total_files} files processed ({files_done_pct:.2f}%)\r')
    std_printer.set_variables({
        'total_files': total_files,
        'files_done': 0,
        'files_done_pct': 0
    })

    if len(images) == 0:
        CLI.exit_error("No images found in {0}".format(input_path))

    task_queue = Queue()
    for image_path in images:
        task_queue.put(image_path)

    threads = []
    for c in range(max_threads):
        t = threading.Thread(target=stat_worker, args=(task_queue, rs, out_dim, std_printer))
        t.start()
        threads.append(t)
        task_queue.put(None)

    task_queue.join()

    for t in threads:
        t.join()

    std_printer.add_line("Data mean {m}, Data variance {v}".format(**{'m': rs.get_meanvalue(), 'v': rs.get_variance()}))

    return rs.get_meanvalue(), rs.get_variance()


def process_path(input_path, output_path, labels_table, num_classes, out_dim, mean, var):
    global std_printer

    images = glob.glob(input_path + "/*.nrrd")
    random.shuffle(images)

    total_files = len(images)
    std_printer.set_format(
        'Converting image to numpy array, {files_done} out of {total_files} files processed' +
        '({batches_written} batches written) ({files_done_pct:.2f}%)\r'
    )
    std_printer.set_variables({
        'total_files': total_files,
        'batches_written': 0,
        'files_done': 0,
        'files_done_pct': 0
    })

    cum_batch_size = 0
    batch_number = 0
    cls = list(0 for p in range(num_classes))
    X = []
    Y = []

    if len(images) == 0:
        CLI.exit_error("No images found in {0}".format(input_path))

    for image_path in images:

        # print("Processing subject {0})".format(record_id))
        record_id = ((os.path.splitext(os.path.basename(image_path))[0]).split('_'))[0]
        label = labels_table[record_id]
        cls[label] = 1

        arr = read_image(image_path, out_dim)
        if arr is None:
            files_done = std_printer.get_variable('files_done') + 1
            std_printer.set_variable('files_done', files_done)
            std_printer.set_variable('files_done_pct', (files_done / total_files) * 100)
            continue

        # Zero center and normalize data
        arr -= mean
        arr /= math.sqrt(abs(var))
        # arr /= abs(var)
        # arr  = np.int16(arr)

        if cum_batch_size + arr.nbytes > batch_max_size:
            # print("Batch size reached, starting a new batch")
            write_batch(X, Y, batch_number, output_path)
            X = []
            Y = []
            cum_batch_size = 0
            batch_number += 1
            batches_written = std_printer.get_variable('batches_written') + 1
            std_printer.set_variable('batches_written', batches_written)

        cum_batch_size += arr.nbytes

        X.append(arr)
        Y.append(cls)

        files_done = std_printer.get_variable('files_done') + 1
        std_printer.set_variable('files_done', files_done)
        std_printer.set_variable('files_done_pct', (files_done / total_files) * 100)

    # print(np.array(X).shape)
    # print("Writing last batch")
    write_batch(X, Y, batch_number, output_path)
    batches_written = std_printer.get_variable('batches_written') + 1
    std_printer.set_variable('batches_written', batches_written)
    std_printer.add_line("Done!")


if len(sys.argv) > 1:
    Config.load(sys.argv[1])

ref_dim = Config.get('reference_dimensions')
segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
batch_max_size = Config.get('batch_max_size')
prefix = Config.get('prefix')

output_dimensions = None
try:
    # This will override the calculate dimensions, note that dimensions will be reduced by trimming from both sides
    output_dimensions = tuple(Config.get('output_dimensions'))
except KeyError:
    pass

num_examples, num_classes, labels_table = get_labels()

scale_factor = 1 / active_shrink_factor

out_dim = (int(ref_dim[0] * scale_factor), int(ref_dim[1] * scale_factor), int(ref_dim[2] * scale_factor))
if output_dimensions is not None:
    out_dim = output_dimensions

print('\nOutput image size is ' + str(out_dim) + '\n')

runs = {
    'train': ['raw_augmented', 'segmented_augmented'],
    'test': ['raw', 'segmented']
}

mode = 1 if segment_enabled else 0
input_path = CLI.get_path('train', runs['train'][mode], active_shrink_factor, prefix=prefix)
mean, var = get_data_stats(input_path, out_dim)

for i in runs:

    print('\nProcessing {0} data\n'.format(i))

    input_type = i

    input_subtype = runs[i][mode]
    input_path = CLI.get_path(i, input_subtype, active_shrink_factor, prefix=prefix)
    output_path = CLI.get_path(i, input_subtype + '_np', active_shrink_factor, prefix=prefix)

    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass

    process_path(input_path, output_path, labels_table, num_classes, out_dim, mean, var)
