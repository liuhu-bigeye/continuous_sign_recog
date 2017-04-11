import sys
import pdb
import h5py
import pickle
import numpy as np
sys.path.insert(0, '/home/liuhu/workspace/journal')
from utils import softmax_np

np.random.seed(1111)

prediction_path = '/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_top_k_2017-04-05_20-02-09/ctc_best_path_63_0.412_0.411_top_10.h5'
output_path = '/'.join(prediction_path.split('/')[:-1]+['soft_alignment_63_0.412_0.411_top_10.h5'])

stride = 4
rec_field = 10

if __name__ == '__main__':
    with open('/var/disk1/RWTH2014/database_2014_combine.pkl') as f:
        db = pickle.load(f)['train']

    df = h5py.File(prediction_path)     # alignment, mask, weight, upsamp_indices, ctc_loss, best_path_loss   
    n_samples = df['mask'].shape[0]

    db_out = {}
    for phase in ['train', 'valid']:
        db_out[phase] = {'prediction_lin':[], 'indices':[], 'mask':[]}

    valid_set_rate = 0.09   # (still split by sentence) may nearby segments split into train set and valid set, however is valuable for whole train set continuity
    phases = ['train', 'valid']

    # [u'ctc_loss', u'mask', u'output_lin', u'top_k_alignment', u'top_k_path_loss', u'upsamp_indices', u'weight']
    for idx in range(n_samples):
        phase = phases[int(np.random.random() < valid_set_rate)]
        if df['ctc_loss'][idx] > 5 or sum(df['weight'][idx])>=1.0:
            if df['ctc_loss'][idx] < 80 and (df['ctc_loss'][idx] > 5 or sum(df['weight'][idx])>=1.0):
                print idx, df['ctc_loss'][idx], sum(df['weight'][idx]),'loss out of bound, neglected'
            continue
        # pdb.set_trace()
        begin_index, end_index = db['begin_index'][idx], db['end_index'][idx]

        prediction_lin = list(df['output_lin'][idx])
        mask = df['mask'][idx]
        upsamp_indices = df['upsamp_indices'][idx]
        upsamp_indices = upsamp_indices[:upsamp_indices.argmax()+1]
        h_len = int(sum(mask))

        assert begin_index + max(upsamp_indices) < end_index
        # pdb.set_trace()
        assert h_len * stride == len(upsamp_indices)

        db_out[phase]['prediction_lin'].extend(prediction_lin[:h_len])

        indices = [[begin_index + upsamp_indices[j] for j in range(max(0, h*stride-3), min(h*stride+7, len(upsamp_indices)))] + [-1]*(10-min(h*stride+7, len(upsamp_indices))+max(0, h*stride-3)) for h in range(h_len)]
        mask = [[int(k>=0 and k<len(upsamp_indices)) for k in range(h*stride-3, h*stride+7)] for h in range(h_len)]

        db_out[phase]['indices'].extend(indices)
        db_out[phase]['mask'].extend(mask)

        if idx%400==0:
            print idx

    df_out = h5py.File(output_path, 'w')
    for phase in ['train', 'valid']:
        df_out.create_dataset('%s_prediction_lin'%phase, data=np.vstack(db_out[phase]['prediction_lin']).astype('float32'))
        df_out.create_dataset('%s_mask'%phase, data=np.vstack(db_out[phase]['mask']).astype('float32'))
        df_out.create_dataset('%s_indices'%phase, data=np.vstack(db_out[phase]['indices']).astype('int32'))
    df_out.close()