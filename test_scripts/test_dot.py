import theano
import theano.tensor as T
import numpy as np
import sys
import pdb

if __name__ == '__main__':
    a_np = np.random.random((2, 25)).astype(np.float32)
    b_np = np.random.random((2, 25, 25)).astype(np.float32)

    a = T.matrix('a')
    b = T.tensor3('b')

    func_sum = theano.function([a, b], T.sum(a[:, :, None] * b[:, :, :], axis=1))
    func_dot = theano.function([a, b], T.batched_dot(a[:, None, :], b)[:, 0])

    import time

    start = time.time()
    answer_s = func_sum(a_np, b_np)
    end = time.time()
    print 'sum:', end-start

    start = time.time()
    answer_d = func_dot(a_np, b_np)
    end = time.time()
    print 'dot:', end-start

    pdb.set_trace()
