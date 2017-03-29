import sys
import pdb
import h5py
import pickle
import numpy as np

np.random.seed(1111)

prediction_path = '/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_2017-03-28_20-41-05/ctc_best_path_63_0.412_0.411.h5'
output_path = prediction_path[:-3]+'.pkl'

stride = 4
rec_field = 10

if __name__ == '__main__':
    with open('/home/trunk/disk1/database-rwth-2014/phoenix2014-release/database_2014_combine.pkl') as f:
        db = pickle.load(f)['train']
    df = h5py.File(prediction_path)     # alignment, mask, weight, upsamp_indices, ctc_loss, best_path_loss

    db_out = {}
    for phase in ['train', 'valid']:
        db_out[phase] = {'token':[], 'weight':[], 'indices':[]}

    n_samples = len(db['token'])
    valid_set_rate = 0.09   # may nearby segments split into train set and valid set, however is valuable for whole train set continuity
    phases = ['train', 'valid']

    for idx in range(n_samples):
        phase = phases[int(np.random.random() < valid_set_rate)]
        for offset in range(4):
            if df['ctc_loss_%d'%offset][idx] > 20 or df['best_path_loss_%d'%offset][idx] > 20:
                print idx, offset, df['ctc_loss_%d'%offset][idx], df['best_path_loss_%d'%offset][idx], 'loss out of bound, neglected'
                continue

            begin_index, end_index = db['begin_index'][idx] + offset, db['end_index'][idx]

            alignment = df['alignment_%d'%offset][idx]
            mask = df['mask_%d'%offset][idx]
            weight = df['weight_%d'%offset][idx]
            upsamp_indices = df['upsamp_indices_%d'%offset][idx]
            upsamp_indices = upsamp_indices[:upsamp_indices.argmax()+1]
            h_len = int(sum(mask))

            assert begin_index + max(upsamp_indices) < end_index
            # pdb.set_trace()
            assert h_len * stride == len(upsamp_indices)

            db_out[phase]['token'].extend(alignment[:h_len])
            db_out[phase]['weight'].extend([weight]*h_len)
            indices = [[begin_index + upsamp_indices[j] for j in range(max(0, h*stride-3), min(h*stride+7, end_index-begin_index))] for h in range(h_len)]
            db_out[phase]['indices'].extend(indices)

        if idx%40==0:
            print idx

    with open(output_path, 'wb') as f:
        pickle.dump(db_out, f)