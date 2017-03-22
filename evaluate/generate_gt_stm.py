import os
import pickle

if __name__ == '__main__':
	pkl_file = '/home/trunk/disk1/database-rwth-2014/phoenix2014-release/database_2014_combine.pkl'
	phase = 'test'

	with open(pkl_file) as f:
		d = pickle.load(f)

	db = d[phase]
	vocabulary = db['vocabulary']

	lines = []
	for i in range(len(db['folder'])):
		folder = db['folder'][i].split('/')[0]
		channel = db['folder'][i].split('/')[1]
		begin_t = '0.0'
		end_t = '1.79769e+308'
		signer = db['signer'][i]
		annotation = ' '.join(db['annotation'][i])

		line = ' '.join([folder, channel, signer, begin_t, end_t, annotation])
		lines.append(line)

	lines = sorted(lines)

	save_file = './phoenix2014-groundtruth-%s.stm' % phase
	with open(save_file, 'w') as f:
		for line in lines:
			f.write(line + '\n')