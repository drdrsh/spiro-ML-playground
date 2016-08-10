import numpy as np
import os, sys, subprocess, time, glob, csv


max_process = 8
min_count   = 50      # Minimum number of replicas per image
max_count   = 100     # Maximum number of replicas per image

files_done  = 0
total_files = 0
processes = []

def get_label_dict(labels_file):
    result = {}
    with open(labels_file, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            record_id = row['record_id']
            result[record_id] = {}
            if row['emphysema'] != '':
                result[record_id]['emphysema'] = int(row['emphysema'])
            else:
                result[record_id]['emphysema'] = 0

            if row['bronchiectasis1'] != '':
                result[record_id]['bronchiectasis'] = int(row['bronchiectasis1'])
            else:
                result[record_id]['bronchiectasis'] = 0
    return result

def update_cli():
    global total_files, files_done
    sys.stdout.write('Performing image augmentation, {0} out of {1} files processed ({2:.2f}%)\r'.format(files_done, total_files,(files_done/total_files) * 100 ))
    sys.stdout.flush()

def wait_for_empty_spot(processes):
    global files_done
    while processes:
        for proc in processes:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                processes.remove(proc)
                files_done += proc.count
                update_cli()
                break
            else: # No process is done, wait a bit and check again.
                time.sleep(.1)
                continue

def wait_for_empty_queue(processes):
    global files_done
    while processes:
        for proc in processes:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                processes.remove(proc)
                files_done += proc.count
                update_cli()
            else: # No process is done, wait a bit and check again.
                time.sleep(.1)



if len(sys.argv) != 3:
    print("Usage: augmenter_runner.py discrete_directory_name output_directory_name")
    sys.exit(1)

try:
    os.mkdir(os.path.abspath(sys.argv[2]))
except:
    pass

labels_table = get_label_dict('../../Data/labels.csv')


files = glob.glob(sys.argv[1] + '/' + "*.nrrd")


# Pre run over the data to estimate class imbalance
dist = []
for file in files:

    record_id = ((os.path.splitext(os.path.basename(file))[0]).split('_'))[0]
    label = labels_table[record_id]

    if label['emphysema']:
        dist.append(1)
    else:
        dist.append(0)

# This will decide how many replicas are created for each file based on its class so that we can acheive class balance
dist= np.array(dist)
bin = np.bincount(dist)
flip = np.abs( (bin / np.sum(bin)) - 1.0 )
additional = max_count - min_count
counts = np.ceil(additional * flip) + min_count

total_files = int(np.sum(bin * counts))

for file in files:

    update_cli()

    record_id = ((os.path.splitext(os.path.basename(file))[0]).split('_'))[0]
    label = labels_table[record_id]['emphysema']

    full_input_path  = os.path.abspath(file)
    full_output_path = os.path.abspath(sys.argv[2] + '/')

    # Augment this image taking into account class imabalnce
    count = int(counts[label])

    if len(processes) < max_process:
        with open(os.devnull, 'w') as fp:
            p = subprocess.Popen(['./ImageAugment', '--input', full_input_path, '--output', full_output_path, '--count', str(count)], stdout=fp)
            p.count = count
            processes.append(p)
    else:
        wait_for_empty_spot(processes)

    wait_for_empty_queue(processes)
