import pdb
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

floatX = theano.config.floatX
intX = 'int32'

from m_ctc_cost import ctc_cost

def m_eye(length, k=0, d_type=floatX):
    out_eye = T.eye(length, dtype=d_type)
    if k == 0:
        return out_eye
    else:
        out = T.zeros_like(out_eye).astype(d_type)
        out = T.set_subtensor(out[:-k, k:], out_eye[:-k, :-k])
        return out

def ctc_equidistant_cost(pred, pred_len, token, max_dist=2.0, blank=0):
    '''
    ctc cost with equidistant prior
    :param pred:        (Time, nb, voca_size+1)
    :param pred_len:    (nb,)
    :param token:       (nb, U)
    :param max_dist:    int or float
    :param blank:       0 for blank index
    '''
    eps = theano.shared(np.float32(1e-35))    
    Time, nb, _ = pred.shape
    U = token.shape[1]
    token_len = T.sum(T.neq(token, -1), axis=-1)    # (nb,)

    max_dist = T.round(max_dist * pred_len / token_len).astype(intX) # (nb)

    # token_with_blank
    token = token[:, :, None]    # (nb, U, 1)
    token_blank = T.ones_like(token, dtype=intX)*blank
    token_with_blank = T.concatenate([token_blank, token, token_blank], axis=-1)    # (nb, U, 3)

    # only use these predictions
    pred = pred[T.arange(Time)[:, None, None, None], T.arange(nb)[None, :, None, None], token_with_blank[None, :, :, :]]    # (Time, nb, U, 3)

    # recurrence relation
    recurrence_relation = m_eye(3) + m_eye(3, k=1)  # (3, 3)
    recurrence_relation = recurrence_relation.astype(floatX)  # (3, 3)

    # seqs equals
    seqs_equals = T.nonzero(T.eq(token[:, :-1, 0], token[:, 1:, 0]))    # (<nb*U, <nb*U)
    seqs_relation = m_eye(Time, k=1)
    seqs_prob_mult = pred[T.arange(Time)[:, None], seqs_equals[0][None, :], seqs_equals[1][None, :]+1, 0].dimshuffle([1, 0])   # (<nb*U, Time)

    # initiate
    alpha = T.zeros((nb, U, Time), dtype=floatX)
    beta = T.zeros((nb, U, Time, 3), dtype=floatX)

    # 1. error for last step, maybe equidistant for last real step is better
    # 2. minor error for seqs equal steps, max dist += 1
    def step_func(p_t, t, alpha, beta):
        beta = T.dot(beta, recurrence_relation) * p_t[:, :, None, :]
        beta = T.set_subtensor(beta[:, :, t, :2], p_t[:, :, :2])

        # last step might output token_last or blank
        last_steps = T.nonzero(T.eq(t, pred_len-1))[0]
        beta = T.set_subtensor(beta[last_steps, :, :, 1], beta[last_steps, :, :, 1] + beta[last_steps, :, :, 2])

        # might be error when consequence same tokens(fixed)
        alpha_trans = T.set_subtensor(alpha[seqs_equals[0], seqs_equals[1], :], T.dot(alpha[seqs_equals[0], seqs_equals[1], :], seqs_relation) * seqs_prob_mult)

        # (nb, U)
        begin_index = T.maximum(0, t - max_dist)    # (nb,)
        begin_mask, _ = theano.scan(lambda x: T.concatenate([T.zeros((x,), dtype=floatX), T.ones((Time-1-x,), dtype=floatX)]), sequences=[begin_index])

        alpha_t = T.concatenate([beta[:, 0, 0, 1][:, None], (alpha_trans[:, :-1, :-1] * beta[:, 1:, 1:, 1] * begin_mask[:, None, :]).sum(axis=2)], axis=1)
        alpha = T.set_subtensor(alpha[:, :, t], alpha_t)
        return alpha, beta

    (alphas, betas), _ = theano.scan(step_func, sequences=[pred, T.arange(Time)], outputs_info=[alpha, beta])
    labels_prob = alphas[-1, T.arange(nb), token_len-1, pred_len-1]     # (nb)
    cost = -T.log(labels_prob + eps)
    return cost

if __name__ == '__main__':
    pred = T.tensor3('pred')
    length = T.ivector('length')
    token = T.imatrix('token')

    # token_th, token_with_blank_th, pred_th, recurrence_relation_th, alphas_th, betas_th, labels_prob_th, 
    cost_equi_th = ctc_equidistant_cost(pred, length, token, max_dist=1.0)
    f_ctc_equi_loss = theano.function([pred, length, token], [cost_equi_th])

    cost_th = ctc_cost(pred, length, token)
    f_ctc_loss = theano.function([pred, length, token], [cost_th])
    # token_th, token_with_blank_th, pred_th, recurrence_relation_th, alphas_th, betas_th, labels_prob_th])

    # (T, nb, voca_size+1)
    pred_np = np.array([[0.5, 0.4, 0.1],[0.3,0.1,0.6],[0.7,0.2,0.1],[0.3,0.5,0.2]]).astype(floatX)[:,None,:]
    # (nb)
    length_np = np.array([4]).astype(intX)
    # (nb, U)
    token_np = np.array([1,2]).astype(intX)[None,:]

    # token_np_out, token_with_blank_np, pred_np_out, recurrence_relation_np, alphas_np, betas_np, labels_prob_np, 
    print f_ctc_equi_loss(pred_np, length_np, token_np)
    print f_ctc_loss(pred_np, length_np, token_np)