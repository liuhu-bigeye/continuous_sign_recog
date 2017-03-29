import sys
import h5py
import pickle
import numpy as np

np.random.seed(1111)

prediction_path = '/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_2017-03-27_22-24-18/ctc_best_path_63_0.412_0.411_modified.h5'
output_path = prediction_path[:-3]+'.pkl'

stride = 4
rec_field = 10

if __name__ == '__main__':
    with open('/home/trunk/disk1/database-rwth-2014/phoenix2014-release/database_2014_combine.pkl') as f:
        db = pickle.load(f)['train']
    df = h5py.File(prediction_path)     # alignment, mask, weight

    db_out = {}
    for phase in ['train', 'valid']:
        db_out[phase] = {'token':[], 'weight':[], 'begin_index':[], 'end_index':[]}

    n_samples = len(db['token'])
    valid_set_rate = 0.09   # may nearby segments split into train set and valid set, however is valuable for whole train set continuity
    phases = ['train', 'valid']

    for idx in range(n_samples):
        phase = phases[int(np.random.random() < valid_set_rate)]
        for offset in range(4):
            begin_index, end_index = db['begin_index'][idx] + offset, db['end_index'][idx]

            alignment = df['alignment_%d'%offset][idx]
            mask = df['mask_%d'%offset][idx]
            weight = df['weight_%d'%offset][idx]

            h_len = int(sum(mask))
            if (end_index - begin_index) % stride != 0:
                h_len -= 1  # neglect last clip when resampled_at_end

            assert begin_index + h_len * 4 <= end_index
            db_out[phase]['token'].extend(alignment[:h_len])
            db_out[phase]['weight'].extend([weight]*h_len)
            db_out[phase]['begin_index'].extend([begin_index + h * 4 for h in range(h_len)])
            db_out[phase]['end_index'].extend([begin_index + h * 4 for h in range(1, h_len + 1)])

            if idx%40==0:
                print offset, idx

    with open(output_path, 'wb') as f:
        pickle.dump(db_out, f)