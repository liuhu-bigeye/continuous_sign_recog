import os
import sys

loadFile = sys.argv[1]

if os.path.basename(loadFile).split('.')[-1] == 'ctm':
	with open(loadFile, 'r') as f:
		lines = [line.strip() for line in f.readlines()]
		lines = sorted(lines, key = lambda x: (x.split(' ')[0], float(x.split(' ')[2])))
elif os.path.basename(loadFile).split('.')[-1] == 'stm':
	with open(loadFile, 'r') as f:
		lines = [line.strip() for line in f.readlines()]
		lines = sorted(lines, key = lambda x: x.split(' ')[0])
else:
	raise ValueError('unsupported file type')

with open(loadFile, 'w+') as f:
	for line in lines:
		f.write(line + '\n')
