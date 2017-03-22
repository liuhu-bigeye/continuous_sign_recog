import os
import pickle

from eval_utils import *

def main():
	out_file = '/home/trunk/disk1/database-rwth-2014/phoenix2014-release/evaluation/example-hypothesis-test.ctm'

	with open(out_file, 'r') as f:
		lines = f.readlines()

	db_file = '/home/trunk/disk1/database-rwth-2014/phoenix2014-release/database_2014_combine.pkl'
	with open(db_file, 'r') as f:
		db = pickle.load(f)
		voc = db['test']['vocabulary']

	hypotheses = {}
	for line in lines:
		splits = line.strip().split(' ')
		ID = '/'.join(splits[: 2])

		if ID not in hypotheses.keys():
			hypotheses[ID] = []
		hypotheses[ID].append(voc.index(splits[-1]) + 1)

	WER = evaluate(hypotheses, voc, '/home/runpeng/workspace/C3D_CTC/evaluate/', 0)
	print WER


if __name__ == '__main__':
	main()

