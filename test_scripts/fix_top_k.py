import pdb
import h5py
import numpy as np
import sys
sys.path.insert(0, '/home/liuhu/workspace/journal')

from utils import softmax_np
from try_ctc.m_ctc_cost import top_k_right_path_cost

import theano
import theano.tensor as T

floatX = theano.config.floatX
intX = np.int32

if __name__ == '__main__':
    pred = T.tensor3('pred')
    mask = T.matrix('mask')
    token = T.imatrix('token')

    cost, argmin_token = top_k_right_path_cost(pred, mask, token, k=10, blank=0)
    f_topk_loss = theano.function([pred, mask, token], [cost, argmin_token])

    df = h5py.File('/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_top_k_2017-04-05_20-02-09/ctc_best_path_63_0.412_0.411_top_10.h5')
    # length = int(sum(df['mask'][0]))
    import pickle
    with open('/var/disk1/RWTH2014/database_2014_combine.pkl') as f:
        db = pickle.load(f)

    for i in range(df['mask'].shape[0]):
        print i
        length = int(sum(df['mask'][i]))
        # (T, nb, voca_size+1)
        pred_np = softmax_np(df['output_lin'][i, :length].astype(floatX))[:, None, :]
        # (nb)
        mask_np = df['mask'][i, :length].astype(floatX)[None, :]
        # (nb, U)
        token_np = np.array(db['train']['token'][i])[None, :].astype(intX) + 1
        cost_np, argmin_token_np = f_topk_loss(pred_np, mask_np, token_np)
        pdb.set_trace()

        # df['top_k_alignment'][i, :length] = argmin_token_np[0]
        # df['top_k_alignment'][i, length:] = -1
    df.close()