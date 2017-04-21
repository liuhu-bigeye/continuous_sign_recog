import sys
import pdb
import h5py
import pickle
import numpy as np

np.random.seed(1111)

prediction_path = '/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_top_k_2017-04-05_20-02-09/ctc_best_path_63_0.412_0.411_top_10.h5'
output_path = '/'.join(prediction_path.split('/')[:-1]+['sum_alignment_63_0.412_0.411_top_10.h5'])

stride = 4
rec_field = 10

if __name__ == '__main__':
    with open('/var/disk1/RWTH2014/database_2014_combine.pkl') as f:
        db = pickle.load(f)['train']
    df = h5py.File(prediction_path)     # alignment, mask, weight, upsamp_indices, ctc_loss, best_path_loss

    db_out = {}
    for phase in ['train', 'valid']:
        db_out[phase] = {'weight':[], 'indices':[], 'mask':[]}

    n_samples = len(db['token'])
    valid_set_rate = 0.09   # may nearby segments split into train set and valid set, however is valuable for whole train set continuity
    phases = ['train', 'valid']

    # [u'ctc_loss', u'mask', u'output_lin', u'top_k_alignment', u'top_k_path_loss', u'upsamp_indices', u'weight']
    for idx in range(n_samples):
        phase = phases[int(np.random.random() < valid_set_rate)]

        begin_index, end_index = db['begin_index'][idx], db['end_index'][idx]
        mask = df['mask'][idx]

        upsamp_indices = df['upsamp_indices'][idx]
        upsamp_indices = upsamp_indices[:upsamp_indices.argmax() + 1]
        h_len = int(sum(mask))

        assert begin_index + max(upsamp_indices) < end_index
        assert h_len * stride == len(upsamp_indices)

        prediction = np.zeros((h_len, 1295+1), dtype=np.float32)
        indices = [[begin_index + upsamp_indices[j] for j in range(max(0, h * stride - 3), min(h * stride + 7, len(upsamp_indices)))] + [-1]*(10-min(h*stride+7, len(upsamp_indices))+max(0, h*stride-3)) for h in range(h_len)]
        mask = [[int(k >= 0 and k < len(upsamp_indices)) for k in range(h * stride - 3, h * stride + 7)] for h in range(h_len)]

        db_out[phase]['indices'].extend(indices)
        db_out[phase]['mask'].extend(mask)
        for k in range(10):
            if df['ctc_loss'][idx] > 5 or df['top_k_path_loss'][idx, k] > 5 or sum(df['weight'][idx])>=1.0:
                if df['ctc_loss'][idx] < 80 and (df['ctc_loss'][idx] > 5 or sum(df['weight'][idx])>=1.0):
                    print idx, k, df['ctc_loss'][idx], df['top_k_path_loss'][idx, k], sum(df['weight'][idx]),'loss out of bound, neglected'
                continue

            alignment = df['top_k_alignment'][idx, :h_len, k]
            weight = df['weight'][idx, k]
            prediction[np.arange(h_len), alignment] += weight

        db_out[phase]['weight'].extend(prediction)

        if idx%400==0:
            print idx

    df_out = h5py.File(output_path, 'w')
    for phase in ['train', 'valid']:
        df_out.create_dataset('%s_weight' % phase, data=np.vstack(db_out[phase]['weight']).astype('float32'))
        df_out.create_dataset('%s_mask' % phase, data=np.vstack(db_out[phase]['mask']).astype('float32'))
        df_out.create_dataset('%s_indices' % phase, data=np.vstack(db_out[phase]['indices']).astype('int32'))
    df_out.close()