import sys
import pdb
import h5py
import pickle
import numpy as np

np.random.seed(1111)

prediction_path = '/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_top_k_2017-04-05_20-02-09/ctc_best_path_63_0.412_0.411_top_10.h5'
output_path = '/'.join(prediction_path.split('/')[:-1]+['soft_alignment.h5'])

stride = 4
rec_field = 10

if __name__ == '__main__':
    with open('/var/disk1/RWTH2014/database_2014_combine.pkl') as f:
        db = pickle.load(f)['train']

    df = h5py.File(prediction_path)     # alignment, mask, weight, upsamp_indices, ctc_loss, best_path_loss
    n_samples, max_h_len, nClasses = df['output_lin'].shape
    
    df_out = h5py.File(output_path)
    df_out.create_dataset('train_prediction', shape=(train_num, nClasses))
    df_out.create_dataset('train_indices', shape=(train_num, 10))
    df_out.create_dataset('train_mask', shape=(train_num, 10))

    df_out.create_dataset('valid_prediction', shape=(valid_num, nClasses))
    df_out.create_dataset('valid_indices', shape=(valid_num, 10))
    df_out.create_dataset('valid_mask', shape=(valid_num, 10))
    
    db_out = {}
    for phase in ['train', 'valid']:
        db_out[phase] = {'prediction':[], 'indices':[], 'mask':[]}

    valid_set_rate = 0.09   # may nearby segments split into train set and valid set, however is valuable for whole train set continuity
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

        prediction = list(softmax_np(df['output_lin'][idx]))
        mask = df['mask'][idx]
        upsamp_indices = df['upsamp_indices'][idx]
        upsamp_indices = upsamp_indices[:upsamp_indices.argmax()+1]
        h_len = int(sum(mask))

        assert begin_index + max(upsamp_indices) < end_index
        # pdb.set_trace()
        assert h_len * stride == len(upsamp_indices)

        db_out[phase]['prediction'].extend(prediction[:h_len])

        indices = [[begin_index + upsamp_indices[j] for j in range(max(0, h*stride-3), min(h*stride+7, len(upsamp_indices)))] for h in range(h_len)]
        mask = [[int(k>=0 and k<len(upsamp_indices)) for k in range(h*stride-3, h*stride+7)] for h in range(h_len)]

        db_out[phase]['indices'].extend(indices)
        db_out[phase]['mask'].extend(mask)

        if idx%400==0:
            print idx

    train_num = len(db_out['train']['prediction'])
    valid_num = len(db_out['valid']['prediction'])

    with open(output_path, 'wb') as f:
        pickle.dump(db_out, f)
