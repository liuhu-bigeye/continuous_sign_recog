#!/usr/bin/env python

import os
import pdb
import glog
import h5py
import numpy as np
np.set_printoptions(precision=4)

import sys
sys.path.insert(0, '/home/liuhu/workspace/journal/')
from loader_config import Config

def valid(model, valid_set, config, epoch):
    losses = np.array([])
    # inputs: feat, token, weight
    # returns: loss_all, loss_classify, acc1, acc5
    for valid_inputs in valid_set.iterate():
        valid_return = model.valid_func(*valid_inputs)
        losses = np.vstack((losses, valid_return)) if losses.shape != (0,) else np.array(valid_return)

    losses = np.mean(losses, axis=0)
    glog.info('%s set, Epoch %d, losses = %s'%(valid_set.phase, epoch, losses))
    return losses[-1]   # acc5


def train_valid(model, train_set, valid_set, config):
    max_acc = -np.inf	# valid_acc
    patienced_epoch = 0
    max_acc_epoch = -1

    for epoch in range(config.items['starting'], config.items['max_epoch']):
        # valid and test
        acc_valid = valid(model, valid_set, config, epoch)

        # dealing with patience
        if acc_valid > max_acc:
            max_acc = acc_valid
            max_acc_epoch = epoch
            patienced_epoch = 0
        else:
            patienced_epoch += 1
            if patienced_epoch > config.items['patience']:
                patienced_epoch = 0
                model.set_learning_rate(np.array(model.learning_rate.get_value() * 0.2, dtype=np.float32))
        glog.info('No acc up for %d epoch, patienced for %d epoch, max_acc: %f' % (epoch - max_acc_epoch, patienced_epoch, max_acc))

        losses = np.array([])
        for iter, train_inputs in enumerate(train_set.iterate()):
            train_return = model.train_func(*train_inputs)
            losses = np.vstack((losses, train_return)) if losses.shape!=(0,) else np.array(train_return)
            if iter % config.items['disp_iter']==0:
                glog.info('Epoch %d, Iteration %d, lr = %.2e, Training loss = %s' %
                          (epoch, iter, model.learning_rate.get_value().astype(np.float32), np.mean(losses, axis=0)))
                model.set_learning_rate()
                losses = np.array([])

        # snapshot
        save_file = os.path.join(config.items['snap_path'], config.items['prefix']+'_epoch_%04d' % epoch)
        glog.info('Snap to path: %s' % save_file)
        model.save_model(save_file)
        model.set_learning_rate()
    glog.info('done, max acc = dev: %f, at epoch %d' % (max_acc, max_acc_epoch))

def main():
    if len(sys.argv) == 3:
        config = Config(sys.argv[1], sys.argv[2])
    else:
        assert False

    phase = config.items['phase']
    from utils import mkdir_safe, log_self
    # log_self(__file__)

    glog.info('generating model...')
    from model import Model
    model = Model(phase=phase, config=config)

    # load model
    if 'model' in config.items.keys():
        glog.info('loading model: %s...' % config.items['model'])
        model.load_model(config.items['model'])
    elif 'model_feat' in config.items.keys():
        glog.info('loading feat model: %s...' % config.items['model_feat'])
        model.load_model_feat(config.items['model_feat'])

    try:
        config.items['starting'] = int(config.items['model'].split('_')[-1])
    except:
        config.items['starting'] = 0

    # snapshot path
    mkdir_safe(config.items['snap_path'])

    # model.make_functions()
    from reader import Reader
    train_set = Reader(phase='train', batch_size=config.items['batch_size'], do_shuffle=True, distortion=True)
    valid_set = Reader(phase='valid', batch_size=10, do_shuffle=False, distortion=False)

    glog.info('ctc training...')
    train_valid(model, train_set, valid_set, config)

    glog.info('end')

if __name__ == '__main__':
    main()