import os, sys, subprocess

if len(sys.argv) != 3:
	print("Usage: timmer_runner.py discrete_directory_name output_directory_name")
	sys.exit(1)

for x in os.walk(sys.argv[1]):

	try:
		os.mkdir(os.path.abspath(sys.argv[2]))
	except:
		pass

	for file in x[2]:
		full_input_path = os.path.abspath(sys.argv[1] + '/' + file)
		full_output_path = os.path.abspath(sys.argv[2] + '/' + file)
		print(full_input_path + " => " + full_output_path)
		subprocess.call(['./LungSegment', full_input_path, full_output_path])