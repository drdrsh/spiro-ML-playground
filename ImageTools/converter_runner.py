import os, sys, subprocess

if len(sys.argv) != 3:
	print("Usage: runner.py series_directory_name output_directory_name")
	sys.exit(1)

for x in os.walk(sys.argv[1]):
	basename = os.path.basename(x[0])
	if basename == '.' or basename == '':
		continue

	full_input_path = os.path.abspath(sys.argv[1] + '/' + basename)
	full_output_path = os.path.abspath(sys.argv[2] + '/' + basename + ".nrrd")

	try:
		os.mkdir(os.path.abspath(sys.argv[2]))
	except:
		pass


	subprocess.call(['./ImageConverter', full_input_path, full_output_path])