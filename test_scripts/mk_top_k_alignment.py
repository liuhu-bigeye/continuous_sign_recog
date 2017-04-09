import sys
import pdb
import h5py
import pickle
import numpy as np

np.random.seed(1111)

prediction_path = '/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_top_k_2017-04-05_20-02-09/ctc_best_path_63_0.412_0.411_top_10.h5'
output_path = prediction_path[:-3]+'.pkl'

stride = 4
rec_field = 10

if __name__ == '__main__':
    with open('/var/disk1/RWTH2014/database_2014_combine.pkl') as f:
        db = pickle.load(f)['train']
    df = h5py.File(prediction_path)     # alignment, mask, weight, upsamp_indices, ctc_loss, best_path_loss

    db_out = {}
    for phase in ['train', 'valid']:
        db_out[phase] = {'token':[], 'weight':[], 'indices':[], 'mask':[]}

    n_samples = len(db['token'])
    valid_set_rate = 0.09   # may nearby segments split into train set and valid set, however is valuable for whole train set continuity
    phases = ['train', 'valid']

    # [u'ctc_loss', u'mask', u'output_lin', u'top_k_alignment', u'top_k_path_loss', u'upsamp_indices', u'weight']
    for idx in range(n_samples):
        phase = phases[int(np.random.random() < valid_set_rate)]
        for k in range(10):
            if df['ctc_loss'][idx] > 5 or df['top_k_path_loss'][idx, k] > 5 or sum(df['weight'][idx])>=1.0:
                if df['ctc_loss'][idx] < 80 and (df['ctc_loss'][idx] > 5 or sum(df['weight'][idx])>=1.0):
                    print idx, k, df['ctc_loss'][idx], df['top_k_path_loss'][idx, k], sum(df['weight'][idx]),'loss out of bound, neglected'
                continue
            # pdb.set_trace()
            begin_index, end_index = db['begin_index'][idx], db['end_index'][idx]

            alignment = df['top_k_alignment'][idx, :, k]
            mask = df['mask'][idx]
            weight = df['weight'][idx, k]
            upsamp_indices = df['upsamp_indices'][idx]
            upsamp_indices = upsamp_indices[:upsamp_indices.argmax()+1]
            h_len = int(sum(mask))

            assert begin_index + max(upsamp_indices) < end_index
            # pdb.set_trace()
            assert h_len * stride == len(upsamp_indices)

            db_out[phase]['token'].extend(alignment[:h_len])
            db_out[phase]['weight'].extend([weight]*h_len)

            indices = [[begin_index + upsamp_indices[j] for j in range(max(0, h*stride-3), min(h*stride+7, len(upsamp_indices)))] for h in range(h_len)]
            mask = [[int(k>=0 and k<len(upsamp_indices)) for k in range(h*stride-3, h*stride+7)] for h in range(h_len)]

            db_out[phase]['indices'].extend(indices)
            db_out[phase]['mask'].extend(mask)

        if idx%400==0:
            print idx

    with open(output_path, 'wb') as f:
        pickle.dump(db_out, f)
