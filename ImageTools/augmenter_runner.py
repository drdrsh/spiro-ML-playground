import os, sys, subprocess, time

max_process = 8

if len(sys.argv) != 3:
    print("Usage: augmenter_runner.py discrete_directory_name output_directory_name")
    sys.exit(1)

processes = []

def waitForEmptySpot(processes):
    while processes:
        for proc in processes:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                processes.remove(proc)
                break
            else: # No process is done, wait a bit and check again.
                time.sleep(.1)
                continue

def waitForEmptyQueue(processes):
    while processes:
        for proc in processes:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                processes.remove(proc)
            else: # No process is done, wait a bit and check again.
                time.sleep(.1)

for x in os.walk(sys.argv[1]):

    try:
        os.mkdir(os.path.abspath(sys.argv[2]))
    except:
        pass

    for file in x[2]:
        full_input_path = os.path.abspath(sys.argv[1] + '/' + file)
        full_output_path = os.path.abspath(sys.argv[2] + '/')
        print("Processing "+ full_input_path )
        if len(processes) < max_process:
            p = subprocess.Popen(['./ImageAugment', '--input', full_input_path, '--output', full_output_path])
            processes.append(p)
        else:
            waitForEmptySpot(processes)

    waitForEmptyQueue(processes)
