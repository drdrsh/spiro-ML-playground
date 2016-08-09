import os, sys, subprocess, time

max_process = 8
count = 50
files_done = 0
total_files = 0

if len(sys.argv) != 3:
    print("Usage: augmenter_runner.py discrete_directory_name output_directory_name")
    sys.exit(1)

processes = []

def update_cli():
    global total_files, files_done
    sys.stdout.write('Performing image augmentation, {0} out of {1} files processed ({2:.2f}%)\r'.format(files_done, total_files,(files_done/total_files) * 100 ))
    sys.stdout.flush()

def waitForEmptySpot(processes):
    global files_done
    while processes:
        for proc in processes:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                processes.remove(proc)
                files_done += 1
                update_cli()
                break
            else: # No process is done, wait a bit and check again.
                time.sleep(.1)
                continue

def waitForEmptyQueue(processes):
    global files_done
    while processes:
        for proc in processes:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                processes.remove(proc)
                files_done += 1
                update_cli()
            else: # No process is done, wait a bit and check again.
                time.sleep(.1)


try:
    os.mkdir(os.path.abspath(sys.argv[2]))
except:
    pass


files = []
for (dirpath, dirnames, filenames) in os.walk(sys.argv[1]):
    files.extend(filenames)
    break


total_files = count * len(files)


for file in files:
    full_input_path = os.path.abspath(sys.argv[1] + '/' + file)
    full_output_path = os.path.abspath(sys.argv[2] + '/')
    if len(processes) < max_process:
        with open(os.devnull, 'w') as fp:
            p = subprocess.Popen(['./ImageAugment', '--input', full_input_path, '--output', full_output_path, '--count', str(count)], stdout=fp)
            processes.append(p)
    else:
        waitForEmptySpot(processes)

    waitForEmptyQueue(processes)
