import numpy as np
np.set_printoptions(precision=4)

import pdb
import glog
import theano
import theano.tensor as T
# from theano_ctc import ctc_cost
import h5py

# from utils import Softmax, softmax_np
floatX = theano.config.floatX
intX = np.int32

import sys
sys.path.insert(0, '/home/liuhu/workspace/journal')
from utils import softmax_np
# sys.path.insert(0, '/home/liuhu/tools/rnn_ctc/nnet')

def m_eye(length, k=0):
    if k == 0:
        return T.eye(length)
    else:
        return T.concatenate([T.concatenate([T.zeros((length-k, k), dtype=floatX), T.eye(length-k)], axis=1), T.zeros((k, length), dtype=floatX)], axis=0)

def ctc_cost(pred, pred_len, token, blank=0):
    '''
    ctc_cost of multi sentences
    :param pred: (T, nb, voca_size+1)                    (4,1,3)
    :param pred_len: (nb,)    pred_len of prediction        (1)
    :param token: (nb, U)    -1 for NIL                    (1,2)
    :param blank: (1)
    :return: ctc_cost
    '''

    eps = theano.shared(np.float32(1e-35))
    Time = pred.shape[0]
    nb, U = token.shape[0], token.shape[1]
    token_len = T.sum(T.neq(token, -1), axis=-1)

    # token_with_blank
    token = token[:, :, None]    # (nb, U, 1)
    token_with_blank = T.concatenate((T.ones_like(token, dtype=intX)*blank, token), axis=2).reshape((nb, 2*U))
    token_with_blank = T.concatenate((token_with_blank, T.ones((nb, 1), dtype=intX)*blank), axis=1)    # (nb, 2*U+1)
    length = token_with_blank.shape[1]

    # only use these predictions
    pred = pred[T.arange(Time)[:, None, None], T.arange(nb)[None, :, None], token_with_blank[None, :, :]]    # (T, nb, 2U+1)

    # recurrence relation
    sec_diag = T.concatenate((T.zeros((nb, 2), dtype=intX), T.neq(token_with_blank[:, :-2], token_with_blank[:, 2:])), axis=1) * T.neq(token_with_blank, blank)    # (nb, 2U+1)
    recurrence_relation = T.tile((m_eye(length) + m_eye(length, k=1)), (nb, 1, 1)) + T.tile(m_eye(length, k=2), (nb, 1, 1))*sec_diag[:, None, :]    # (nb, 2U+1, 2U+1)
    recurrence_relation = recurrence_relation.astype(floatX)

    # alpha
    alpha = T.zeros_like(token_with_blank, dtype=floatX)
    alpha = T.set_subtensor(alpha[:, :2], pred[0, :, :2])################(nb, 2U+1)

    # dynamic programming
    # (T, nb, 2U+1)
    probability, _ = theano.scan(lambda curr, accum: T.sum(accum[:, :, None] * recurrence_relation, axis=1) * curr, sequences=[pred[1:]], outputs_info=[alpha])
    # T.batched_dot(accum[:, None, :], recurrence_relation)[:, 0] * curr,

    labels_2 = probability[pred_len-2, T.arange(nb), 2*token_len-1]
    labels_1 = probability[pred_len-2, T.arange(nb), 2*token_len]
    labels_prob = labels_2 + labels_1

    cost = -T.log(labels_prob+eps)
    return cost
    # return token_with_blank, pred, sec_diag, recurrence_relation, alpha, probability, cost

def best_right_path_cost(pred, mask, token, blank=0):
    '''
    best right path cost of multi sentences
    :param pred: (T, nb, voca_size+1)                    (4,1,3)
    :param mask: (nb, T)
    # :param pred_len: (nb,)    pred_len of prediction        (1)
    :param token: (nb, U)    -1 for NIL                    (1,2)
    :param blank: (1)

    :return: best_right_path_cost (nb,)
    :return: argmin_token (nb, T) best path, -1 for null
    '''

    pred_len = mask.sum(axis=-1).astype('int32')
    eps = theano.shared(np.float32(1e-35))
    EPS = theano.shared(np.float32(35))

    t = pred.shape[0]
    nb, U = token.shape[0], token.shape[1]
    token_len = T.sum(T.neq(token, -1), axis=-1)

    # token_with_blank
    token = token[:, :, None]    # (nb, U, 1)
    token_with_blank = T.concatenate((T.ones_like(token, dtype=intX)*blank, token), axis=2).reshape((nb, 2*U))
    token_with_blank = T.concatenate((token_with_blank, T.ones((nb, 1), dtype=intX)*blank), axis=1)    # (nb, 2*U+1)
    length = token_with_blank.shape[1]

    # only use these predictions
    pred = pred[:, T.tile(T.arange(nb), (length, 1)).T, token_with_blank]    # (T, nb, 2U+1)
    pred = -T.log(pred + eps)

    # recurrence relation
    sec_diag = T.concatenate((T.zeros((nb, 2), dtype=intX), T.neq(token_with_blank[:, :-2], token_with_blank[:, 2:])), axis=1) * T.neq(token_with_blank, blank)    # (nb, 2U+1)
    recurrence_relation = T.tile((m_eye(length) + m_eye(length, k=1)), (nb, 1, 1)) + T.tile(m_eye(length, k=2), (nb, 1, 1))*sec_diag[:, None, :]    # (nb, 2U+1, 2U+1)
    recurrence_relation = -T.log(recurrence_relation + eps).astype(floatX)

    # alpha
    alpha = T.ones_like(token_with_blank, dtype=floatX) * EPS
    alpha = T.set_subtensor(alpha[:, :2], pred[0, :, :2])################(nb, 2U+1)

    # dynamic programming
    # (T, nb, 2U+1)
    [log_probability, argmin_pos_1], _ = theano.scan(lambda curr, accum: ((accum[:, :, None] + recurrence_relation).min(axis=1) + curr, (accum[:, :, None] + recurrence_relation).argmin(axis=1)),
                                                   sequences=[pred[1:]], outputs_info=[alpha, None])

    # why pred_len-2?
    labels_1 = log_probability[pred_len-2, T.arange(nb), 2*token_len-1]		# (nb,)
    labels_2 = log_probability[pred_len-2, T.arange(nb), 2*token_len]		# (nb,)
    concat_labels = T.concatenate([labels_1[:, None], labels_2[:, None]], axis=-1)
    argmin_labels = concat_labels.argmin(axis=-1)

    cost = concat_labels.min(axis=-1)

    min_path = T.ones((t-1, nb), dtype=intX)*-1  # -1 for null
    min_path = T.set_subtensor(min_path[pred_len-2, T.arange(nb)], 2*token_len-1+argmin_labels)

    # (T-1, nb)
    min_full_path, _ = theano.scan(lambda m_path, argm_pos, m_full_path: argm_pos[T.arange(nb), T.maximum(m_path, m_full_path).astype('int32')].astype('int32'),
                                   sequences=[min_path[::-1], argmin_pos_1[::-1]], outputs_info=[min_path[-1]])
    argmin_pos = T.concatenate((min_full_path[::-1], min_path[-1][None, :]), axis=0) # (T, nb)
    argmin_pos = T.set_subtensor(argmin_pos[pred_len-1, T.arange(nb)], 2*token_len-1+argmin_labels)

    argmin_token = token_with_blank[T.arange(nb)[None, :], argmin_pos]

    # (nb,), (nb, T)
    return cost, (argmin_token.transpose((1, 0))*mask + mask - 1).astype('int32')# alpha, log_probability, argmin_pos_1, argmin_labels, min_path, min_full_path, argmin_pos, token_with_blank, argmin_token

def top_k_right_path_cost(pred, mask, token, k, blank=0):
    '''
    best right path cost of multi sentences
    :param pred: (T, nb, voca_size+1)                    (4,1,3)
    :param mask: (nb, T)
    :param token: (nb, U)    -1 for NIL                    (1,2)
    :param k:     (1) top k paths
    :param blank: (1)

    :return: top_k_path_cost (nb, k)
    :return: argmin_k_token (nb, k, T) top k path, -1 for null
    '''

    pred_len = mask.sum(axis=-1).astype('int32')
    eps = theano.shared(np.float32(1e-35))
    EPS = theano.shared(np.float32(35))

    t = pred.shape[0]
    nb, U = token.shape[0], token.shape[1]
    token_len = T.sum(T.neq(token, -1), axis=-1)

    # token_with_blank
    token = token[:, :, None]    # (nb, U, 1)
    token_with_blank = T.concatenate((T.ones_like(token, dtype=intX)*blank, token), axis=2).reshape((nb, 2*U))
    token_with_blank = T.concatenate((token_with_blank, T.ones((nb, 1), dtype=intX)*blank), axis=1)    # (nb, 2*U+1)
    length = token_with_blank.shape[1]

    # only use these predictions
    pred = pred[:, T.tile(T.arange(nb), (length, 1)).T, token_with_blank]    # (T, nb, 2U+1)
    pred = -T.log(pred + eps)

    # recurrence relation
    sec_diag = T.concatenate((T.zeros((nb, 2), dtype=intX), T.neq(token_with_blank[:, :-2], token_with_blank[:, 2:])), axis=1) * T.neq(token_with_blank, blank)    # (nb, 2U+1)
    recurrence_relation = T.tile((m_eye(length) + m_eye(length, k=1)), (nb, 1, 1)) + T.tile(m_eye(length, k=2), (nb, 1, 1))*sec_diag[:, None, :]    # (nb, 2U+1, 2U+1)
    recurrence_relation = -T.log(recurrence_relation + eps).astype(floatX)

    # alpha
    alpha = T.ones((nb, k, length), dtype=floatX) * EPS
    alpha = T.set_subtensor(alpha[:, 0, :2], pred[0, :, :2])        #(nb, k, 2U+1)

    def step_func_1(curr, accum):
        '''
        :param curr: (nb, length)
        :param accum: (nb, k, length)
        '''
        alpha_t = (accum[:, :, :, None] + recurrence_relation[:, None, :, :]).reshape((nb, k*length, length))
        accum_t = alpha_t.sort(axis=1)[:, :k, :] + curr[:, None, :]
        argmin_k_t = alpha_t.argsort(axis=1)[:, :k, :]      # from 0 to k*length
        return accum_t, argmin_k_t

    # dynamic programming
    # (T-1, nb, k, length),   (T-1, nb, k, length)
    [log_probability, argmin_pos_k], _ = theano.scan(step_func_1, sequences=[pred[1:]], outputs_info=[alpha, None])

    labels_1 = log_probability[(pred_len-2)[:, None], T.arange(nb)[:, None], T.arange(k)[None, :], (2*token_len-1)[:, None]]     # (nb, k)
    labels_2 = log_probability[(pred_len-2)[:, None], T.arange(nb)[:, None], T.arange(k)[None, :], (2*token_len)[:, None]]       # (nb, k)
    concat_labels = T.concatenate([labels_1, labels_2], axis=-1)
    argmin_labels = (2*token_len-1)[:, None] + concat_labels.argsort(axis=-1)[:, :k].astype('int32') / k       # (nb, k) from 0 to 2k
    cost = concat_labels.sort(axis=-1)[:, :k]

    min_path = T.ones((t-1, nb, k), dtype=intX)*-1  # (T-1, nb, k) -1 for null
    min_path = T.set_subtensor(min_path[(pred_len-2)[:, None], T.arange(nb)[:, None], T.arange(k)[None, :]], argmin_labels + T.arange(k)[None, :] * length)    # set (nb, k)

    def step_func_2(m_path, argm_pos, m_full_path):
        '''
        :param m_path: (nb, k) min path (from 0 to k*length)
        :param argm_pos: (nb, k, length) argmin_pos_k
        :param m_full_path: (nb, k) min full path (from 0 to k*length)
        '''
        path_here = T.maximum(m_path, m_full_path).astype('int32')                                          # (nb, k)
        m_full_return = argm_pos.reshape((nb, k*length))[T.arange(nb)[:, None], path_here].astype('int32')  # (nb, k)
        return m_full_return

    # (T-1, nb, k)
    min_full_path, _ = theano.scan(step_func_2, sequences=[min_path[::-1], argmin_pos_k[::-1]], outputs_info=[min_path[-1]])
    # (T, nb, k)
    argmin_pos = T.concatenate((min_full_path[::-1], min_path[-1][None, :, :]), axis=0)                        # (T, nb, k)
    argmin_pos = T.set_subtensor(argmin_pos[(pred_len-1)[:, None], T.arange(nb)[:, None], T.arange(k)[None, :]], argmin_labels + T.arange(k)[None, :] * length)

    # (nb, k*length) -> (T, nb, k)
    argmin_token = T.tile(token_with_blank[:, None, :], (1, k, 1)).reshape((nb, k*length))[T.arange(nb)[None, :, None], argmin_pos]

    mask_k = T.le(cost, EPS-1)
    argmin_token = (argmin_token.transpose((1, 0, 2)) * mask[:, :, None] + mask[:, :, None] - 1) * mask_k[:, None, :] + mask_k[:, None, :] - 1

    # (nb, k), (nb, T, k)
    return cost, argmin_token.astype('int32')#, log_probability, argmin_pos_k, min_full_path

def greedy_cost(pred, mask):
    '''
    greedy loss of multiple sentences
    :param pred: (T, nb, voca_size+1)
    :param mask: (nb, T)
    :return: greedy_cost (nb,)
    '''
    greedy_pred = pred.max(axis=-1)
    greedy_pred = T.maximum(greedy_pred, 1 - mask.dimshuffle((1, 0)))
    log_greedy_pred = -T.log(greedy_pred)

    greedy_cost = log_greedy_pred.sum(axis=0)
    return greedy_cost

if __name__ == '__main__':
    pred = T.tensor3('pred')
    length = T.ivector('length')
    token = T.imatrix('token')

    token_with_blank_th, pred_th, sec_diag_th, recurrence_relation_th, alpha_th, probability_th, cost_th = ctc_cost(pred, length, token, blank=0)
    f_ctc_loss = theano.function([pred, length, token], [token_with_blank_th, pred_th, sec_diag_th, recurrence_relation_th, alpha_th, probability_th, cost_th], on_unused_input='warn')

    # (T, nb, voca_size+1)
    pred_np = np.array([[0.5, 0.4, 0.1],[0.3,0.1,0.6],[0.7,0.2,0.1],[0.3,0.5,0.2]]).astype(floatX)[:,None,:]
    # (nb)
    length_np = np.array([4]).astype(intX)
    # (nb, U)
    token_np = np.array([2,1]).astype(intX)[None,:]

    token_with_blank_np, pred_np, sec_diag_np, recurrence_relation_np, alpha_np, probability_np, cost_np = f_ctc_loss(pred_np, length_np, token_np)
    pdb.set_trace()
    # glog.info('%s, %s, %s'%(pred_np.shape, length_np.shape, token_np.shape))
    # glog.info(f_ctc_loss(pred_np, length_np, token_np))

# def generate_data(T, nb, length_max, voca_size):
#     # generate preds(nb, T, voca_size), tokens(no 0, -1 for null), lengths(length for preds)

#     preds = softmax_np(np.random.random((T, nb, voca_size+1)))            # (T, nb, voca_size + 1)
#     assert length_max<=T and length_max>2
#     length = np.random.randint(2, length_max+1, size=(nb))                # (nb)
#     tokens = np.array([np.concatenate([np.random.randint(voca_size, size=l), -np.ones(length_max-l)]) for l in length])

#     pred_len = np.zeros((nb))
#     mask = np.zeros((nb, T))
#     for i in range(nb):
#         pred_len[i] = np.random.randint(length[i], T+1)
#         mask[i, :pred_len[i]] = 1
#     return preds.astype(floatX), tokens.astype(intX), pred_len.astype(intX), mask.astype(bool)

# if __name__ == '__main__':
#     pred = T.tensor3('pred')
#     length = T.ivector('length')
#     token = T.imatrix('token')

#     cost, alpha, log_probability, argmin_pos_1, argmin_labels, min_path, min_full_path, argmin_pos, token_with_blank, argmin_token = best_right_path_cost(pred, length, token, blank=0)
#     f_best_loss = theano.function([pred, length, token],
#                                   [cost, alpha, log_probability, argmin_pos_1, argmin_labels, min_path, min_full_path, argmin_pos, token_with_blank, argmin_token])

#     # (T, nb, voca_size+1)
#     pred_np = np.array([[0.5, 0.4, 0.1],[0.3,0.1,0.6],[0.7,0.2,0.1],[0.3,0.5,0.2]]).astype(floatX)[:,None,:]
#     # (nb)
#     length_np = np.array([4]).astype(intX)
#     # (nb, U)
#     token_np = np.array([2,1]).astype(intX)[None,:]

#     glog.info('%s, %s, %s'%(pred_np.shape, length_np.shape, token_np.shape))
#     glog.info('%s'%zip('cost, alpha, log_probability, argmin_pos_1, argmin_labels, min_path, min_full_path, argmin_pos, token_with_blank, argmin_token'.split(', '),
#                        f_best_loss(pred_np, length_np, token_np)))
#     glog.info('cost_gt: %f'%( -np.log(0.105)))

# if __name__ == '__main__':
#     pred = T.tensor3('pred')
#     mask = T.matrix('mask')
#     token = T.imatrix('token')

#     cost, argmin_token = top_k_right_path_cost(pred, mask, token, k=15, blank=0)
#     f_topk_loss = theano.function([pred, mask, token], [cost, argmin_token])

#     # (T, nb, voca_size+1)
#     pred_np = np.array([[0.5, 0.4, 0.1],[0.3,0.1,0.6],[0.7,0.2,0.1],[0.3,0.5,0.2]]).astype(floatX)[:,None,:]
#     # (nb)
#     mask_np = np.array([[1, 1, 1, 1]]).astype(floatX)
#     # (nb, U)
#     token_np = np.array([1, 2]).astype(intX)[None,:]

#     glog.info('cost_gt: %f'%( -np.log(0.105)))
#     cost_np, argmin_token_np = f_topk_loss(pred_np, mask_np, token_np)
#     print cost_np
#     print argmin_token_np
#     pdb.set_trace()

# top_k path from prediction
# if __name__ == '__main__':
#     pred = T.tensor3('pred')
#     mask = T.matrix('mask')
#     token = T.imatrix('token')
#
#     cost, argmin_token, log_probability, argmin_pos_k, min_full_path = top_k_right_path_cost(pred, mask, token, k=15, blank=0)
#     f_topk_loss = theano.function([pred, mask, token], [cost, argmin_token, log_probability, argmin_pos_k, min_full_path])
#
#     df = h5py.File('/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_top_k_2017-04-05_20-02-09/ctc_best_path_63_0.412_0.411_top_10.h5')
#     # length = int(sum(df['mask'][0]))
#     import pickle
#     with open('/var/disk1/RWTH2014/database_2014_combine.pkl') as f:
#         db = pickle.load(f)
#
#     # (T, nb, voca_size+1)
#     pred_np = softmax_np(df['output_lin'][:4].astype(floatX)).transpose([1, 0, 2])
#     # (nb)
#     mask_np = df['mask'][:4].astype(floatX)
#     # (nb, U)
#     token_np = np.vstack([np.concatenate([db['train']['token'][i], np.ones((12-len(db['train']['token'][i])))*-2]) for i in range(4)]).astype(intX) + 1
#
#     # glog.info('cost_gt: %f'%( -np.log(0.105)))
#     cost_np, argmin_token_np, log_probability_np, argmin_pos_k_np, min_full_path_np = f_topk_loss(pred_np, mask_np, token_np)
#     print cost_np
#     print argmin_token_np
#     pdb.set_trace()


# if __name__ == '__main__':
#     pred = T.tensor3('pred')
#     token = T.imatrix('token')
#     mask = T.imatrix('mask')
#     length = T.ivector('length')

#     # ctc_loss = ctc_cost(T.log(pred), token, length)
#     cost = ctc_cost(pred, length, token, blank=0)
#     best_loss, pred_, recurrence_relation, alpha, log_probability = best_right_path_cost(pred, length, token, blank=0)
#     greedy_loss = greedy_cost(pred, mask)

#     f_best_loss = theano.function(inputs=[pred, token, length], outputs=[best_loss, pred_, recurrence_relation, alpha, log_probability])
#     f_m_ctc = theano.function(inputs=[pred, token, length], outputs=[cost])
#     f_greedy = theano.function(inputs=[pred, mask], outputs=[greedy_loss])

#     nb = 5
#     T = 30
#     length_max = 10
#     voca_size = 50

#     for i in range(5):
#         preds, tokens, length, masks = generate_data(T, nb, length_max, voca_size)
#         glog.info('start testing...')
#         glog.info('%s'%(f_m_ctc(preds, tokens, length)))
#         glog.info('%s'%(f_best_loss(preds, tokens, length)[0]))
#         glog.info('%s'%(f_greedy(preds, masks)))
#         cost, pred, recurrence_relation, alpha, log_probability = f_best_loss(preds, tokens, length)
#         pdb.set_trace()
#         print cost.shape, pred.shape, recurrence_relation.shape, alpha.shape, log_probability.shape







