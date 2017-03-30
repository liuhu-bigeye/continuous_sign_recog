import os
import glog
import pickle
import theano
import numpy as np
import theano.tensor as T
import lasagne

from lasagne.layers import *
from lasagne.nonlinearities import *
# import caffe

from try_ctc.m_ctc_cost import ctc_cost, best_right_path_cost
from utils import Softmax

class Model(object):
    def __init__(self, phase, config, vocabulary_size=1295, hidden_ndim=512):
        # need to be same voca_size and hidde_ndim so as to load same shape params
        # self.log_self()
        size = 101
        # model paras
        self.config = config
        self.alpha = np.array(1e-3, dtype=np.float32)
        self.eps = np.array(1e-6, dtype=np.float32)
        self.learning_rate = theano.shared(np.float32(config.items['lr']))

        self.nClasses = vocabulary_size + 1
        self.vocabulary_size = vocabulary_size

        # variables
        data = T.tensor4('data')  # (3*nb, 3, 96, 96) or (nb*len, 3, 96, 96)
        token = T.ivector('token')  # (nb, max_vlen)
        weight = T.vector('weight')
        mask = T.tensor3('mask')
        self.nb = weight.shape[0]

        # label = T.imatrix('label')  # (nb, voca_size)

        net = {}
        # feature extraction
        net['input'] = InputLayer(shape=(None, 3, size, size))  # (3*nb, 3, 96, 96)

        net['conv1'] = Conv2DLayer(incoming=net['input'], num_filters=96, filter_size=7, stride=2)
        net['norm1'] = LocalResponseNormalization2DLayer(incoming=net['conv1'])
        net['pool1'] = MaxPool2DLayer(incoming=net['norm1'], pool_size=3)

        net['conv2'] = Conv2DLayer(incoming=net['pool1'], num_filters=256, filter_size=5)
        net['pool2'] = MaxPool2DLayer(incoming=net['conv2'], pool_size=2)

        net['conv3'] = Conv2DLayer(incoming=net['pool2'], num_filters=512, filter_size=3, pad=1)
        net['conv4'] = Conv2DLayer(incoming=net['conv3'], num_filters=512, filter_size=3, pad=1)
        net['conv5'] = Conv2DLayer(incoming=net['conv4'], num_filters=512, filter_size=3, pad=1)
        net['pool5'] = MaxPool2DLayer(incoming=net['conv5'], pool_size=3) # (3*nb, 512, 3, 3)

        net['fc6'] = DenseLayer(incoming=net['pool5'], num_units=1024)  # auto flatten all the trailing axes
        net['drop6'] = DropoutLayer(incoming=net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(incoming=net['drop6'], num_units=256, nonlinearity=identity)  # (3*nb, 256)

        # encoding network for image features
        net['mask'] = InputLayer(shape=(None, None, None), name='mask')  # (nb, mask_len)

        net['pre_conv1d'] = DimshuffleLayer(ReshapeLayer(incoming=NonlinearityLayer(net['fc6'], nonlinearity=rectify), shape=(self.nb, -1, 1024)), (0, 2, 1))
        net['conv1d_1'] = Conv1DLayer(net['pre_conv1d'], num_filters=1024, filter_size=3, pad='valid')
        net['pool1d_1'] = MaxPool1DLayer(net['conv1d_1'], pool_size=2)	#(nb, 1024, max_hlen)
        net['drop1d_1'] = DropoutLayer(net['pool1d_1'], p=0.5, shared_axes=(2,))
        net['masked_drop1d_1'] = ElemwiseMergeLayer([net['drop1d_1'], net['mask']], merge_function=T.mul)

        net['conv1d_2'] = Conv1DLayer(net['masked_drop1d_1'], num_filters=1024, filter_size=3, pad='valid')
        net['pool1d_2'] = MaxPool1DLayer(net['conv1d_2'], pool_size=2)	#(nb, 1024, max_hlen)
        net['drop1d_2'] = DropoutLayer(net['pool1d_2'], p=0.5, shared_axes=(2,))
        net['classify'] = DenseLayer(ReshapeLayer(net['drop1d_2'], shape=(-1, 1024)), self.nClasses, nonlinearity=softmax)

        # LSTM
        net['lstm_input'] = InputLayer(shape=(None, None, 1024), name='lstm_input')
        net['lstm_frw'] = LSTMLayer(incoming=net['lstm_input'], mask_input=net['mask'], forgetgate=Gate(b=lasagne.init.Constant(1.0)), num_units=hidden_ndim)  # (nb, max_hlen, hidden_ndim)
        net['lstm_bck'] = LSTMLayer(incoming=net['lstm_input'], mask_input=net['mask'], forgetgate=Gate(b=lasagne.init.Constant(1.0)), num_units=hidden_ndim, backwards=True)

        net['lstm_shp'] = ReshapeLayer(ConcatLayer((net['lstm_frw'], net['lstm_bck']), axis=2), shape=(-1, 2*hidden_ndim))  # (nb*max_hlen, 2*hidden_ndim)
        net['out'] = DenseLayer(net['lstm_shp'], self.nClasses, nonlinearity=identity)  # (nb*max_hlen, nClasses)
        net['out_lin'] = ReshapeLayer(net['out'], shape=(self.nb, -1, self.nClasses))

        # # WSDD
        # net['wsdd_input'] = InputLayer(shape=(None, 1024, None), name='wsdd_input')  # (nb, 1024, max_hlen+2)
        # net['wsdd1'] = Conv1DLayer(net['wsdd_input'], num_filters=256, filter_size=2, pad='valid')
        # net['wsdd1_drop'] = DropoutLayer(net['wsdd1'], p=0.5)  # (nb, 256, max_hlen+1)
        # net['wsdd2'] = Conv1DLayer(net['wsdd1_drop'], num_filters=256, filter_size=2, pad='valid')
        # net['wsdd2_drop'] = DropoutLayer(net['wsdd2'], p=0.5)  # (nb, 256, max_hlen)
        #
        # net['predet1a'] = DimshuffleLayer(net['wsdd2_drop'], (0, 2, 1))
        # net['predet1b'] = ReshapeLayer(net['predet1a'], (-1, 256))  # (nb*max_hlen, 256)
        #
        # net['det'] = DenseLayer(net['predet1b'], vocabulary_size, nonlinearity=identity)  # (nb*max_hlen, voca_size)
        # net['det_lin'] = ReshapeLayer(net['det'], (self.nb, -1, vocabulary_size))  # (nb, max_hlen, voca_size)

        self.net = net

        # try save load model
        dummy_save_file = 'dummy.pkl'
        glog.info('try save load dummy model to: %s...' % dummy_save_file)
        self.save_model(dummy_save_file)
        self.load_model(dummy_save_file)
        os.system('rm -rf dummy.pkl')
        glog.info('dummy save load success, remove it and start calculate outputs...')

        if phase == 'cnn_training':
            self.params_full = lasagne.layers.get_all_params(self.net['classify'], trainable=True)
            self.regular_params = lasagne.layers.get_all_params(self.net['classify'], regularizable=True)
            regular_full = lasagne.regularization.apply_penalty(self.regular_params, lasagne.regularization.l2) * np.array(5e-4/2, dtype=np.float32)

            classify_loss_train, classify_acc1_train, classify_acc5_train = self.get_classify_loss(data, token, weight, mask, deterministic=False)
            classify_loss_valid, classify_acc1_valid, classify_acc5_valid = self.get_classify_loss(data, token, weight, mask, deterministic=True)

            loss_train_full = regular_full + classify_loss_train.mean()
            loss_valid_full = regular_full + classify_loss_valid.mean()

            updates = lasagne.updates.adam(loss_train_full, self.params_full, learning_rate=self.learning_rate)
            self.train_func = theano.function([data, token, weight, mask], [loss_train_full, classify_loss_train.mean(), classify_acc1_train.mean(), classify_acc5_train.mean()], updates=updates)
            self.valid_func = theano.function([data, token, weight, mask], [loss_valid_full, classify_loss_valid.mean(), classify_acc1_valid.mean(), classify_acc5_valid.mean()])

        elif phase == 'pretrain':
            pass
            # # for triplet pretrain use
            # self.params_feat = get_all_params(net['fc7'])
            # regular_feat = lasagne.regularization.apply_penalty(self.params_feat, lasagne.regularization.l2) * np.array(5e-4 / 2, dtype=np.float32)
            #
            # ## triplet train loss
            # triplet_loss_train = self.get_triplet_loss(data, deterministic=False)
            # loss_train_feat = triplet_loss_train + regular_feat
            #
            # ## triplet valid loss
            # triplet_loss_valid = self.get_triplet_loss(data, deterministic=True)
            # loss_valid_feat = triplet_loss_valid + regular_feat
            #
            # self.updates = lasagne.updates.momentum(loss_train_feat, self.params_feat, learning_rate=learning_rate, momentum=0.9)
            # self.inputs = [data]
            # self.train_outputs = [loss_train_feat, triplet_loss_train]
            # self.valid_outputs = [loss_valid_feat, triplet_loss_valid]
        # elif phase == 'ctc':
        #     # for ctc loss
        #     self.params_full = lasagne.layers.get_all_params([self.net['drop1d_2'], self.net['out_lin']], trainable=True)
        #     self.regular_params = lasagne.layers.get_all_params([self.net['drop1d_2'], self.net['out_lin']], regularizable=True)
        #     regular_full = lasagne.regularization.apply_penalty(self.regular_params, lasagne.regularization.l2) * np.array(5e-4/2, dtype=np.float32)
        #
        #     # full train loss
        #     ctc_loss_train, pred_train = self.get_ctc_loss(data, mask, token, deteministic=False)
        #     loss_train_full = ctc_loss_train + regular_full
        #
        #     # full valid loss
        #     ctc_loss_valid, pred_valid = self.get_ctc_loss(data, mask, token, deteministic=True)
        #     loss_valid_full = ctc_loss_valid + regular_full
        #
        #     self.updates = lasagne.updates.adam(loss_train_full, self.params_full, learning_rate=self.learning_rate)
        #     self.inputs = [data, mask, token]
        #     self.train_outputs = [loss_train_full, ctc_loss_train, pred_train]
        #     self.valid_outputs = [loss_valid_full, ctc_loss_valid, pred_valid]
        # elif phase == 'extract_feature':
        #     # for feature extraction
        #     fc6 = get_output(self.net['fc6'], data, deterministic = True)
        #     self.feature_func = theano.function(inputs=[data], outputs=fc6)
        # elif phase == 'get_prediction':
        #     embeding = get_output(self.net['drop1d_2'], data, deterministic=True)  # (nb, 1024, len_m)
        #     output_lin = get_output(self.net['out_lin'], {self.net['lstm_input']: T.transpose(embeding, (0, 2, 1)), self.net['mask']: mask}, deterministic=True)
        #
        #     output_softmax = Softmax(output_lin)  # (nb, max_hlen, nClasses)
        #     output_trans = T.transpose(output_softmax, (1, 0, 2))  # (max_hlen, nb, nClasses)
        #
        #     best_path_loss, best_path = best_right_path_cost(output_trans, mask, token)
        #     ctc_loss = ctc_cost(output_trans, T.sum(mask, axis=1, dtype='int32'), token)
        #
        #     # (nb, max_hlen, voca_size+1)
        #     self.predict_func = theano.function(inputs=[data, mask, token], outputs=[best_path_loss, best_path, ctc_loss])

        glog.info('Model built, phase = %s'%phase)
    # def make_functions(self):
    #     self.train_func = theano.function(inputs=self.inputs, outputs=self.train_outputs, updates=self.updates)
    #     self.valid_func = theano.function(inputs=self.inputs, outputs=self.valid_outputs)

    def get_triplet_loss(self, data, deterministic=False):
        fc7 = get_output(self.net['fc7'], data, deterministic=deterministic)  # (3, nb, 256)
        reshape = T.reshape(T.tanh(fc7), newshape=(3, -1, 256))
        anchor = reshape[0]  # (nb, 256)
        positive = reshape[1]
        negative = reshape[2]

        norm_pos = T.pow(T.sum(T.pow(positive - anchor, 2.0), axis=1) + self.eps, 0.5)  # (nb, )
        norm_neg1 = T.pow(T.sum(T.pow(negative - anchor, 2.0), axis=1) + self.eps, 0.5)  # (nb, )
        norm_neg2 = T.pow(T.sum(T.pow(negative - positive, 2.0), axis=1) + self.eps, 0.5)  # (nb, )
        norm_neg = T.min([norm_neg1, norm_neg2], axis=0)

        max_norm = T.max([norm_pos, norm_neg], axis=0)

        d_pos = T.maximum(T.exp(norm_pos - max_norm) / (T.exp(norm_pos - max_norm) + T.exp(norm_neg - max_norm)), self.alpha)
        loss = T.mean(d_pos ** 2)
        return loss#, T.mean(norm_pos), T.mean(norm_neg1), T.mean(norm_neg2), T.mean(norm_neg), T.mean(max_norm)

    def get_ctc_loss(self, data, mask, token, deteministic=False):
        embeding = get_output(self.net['drop1d_2'], data, deterministic=deteministic)

        # loss calculation
        ## ctc loss
        output_lin = get_output(self.net['out_lin'], {self.net['lstm_input']: T.transpose(embeding, (0, 2, 1)), self.net['mask']: mask})
        output_softmax = Softmax(output_lin)  # (nb, max_hlen, nClasses)
        pred = T.argmax(output_softmax, axis=2)

        output_trans = T.transpose(output_softmax, (1, 0, 2))  # (max_hlen, nb, nClasses)
        ctc_loss = ctc_cost(output_trans, T.sum(mask, axis=1, dtype='int32'), token).mean()

        return ctc_loss, pred

    def get_classify_loss(self, data, token, weight, mask, deterministic=False):
        pred = get_output(self.net['classify'], {self.net['input']: data, self.net['mask']: mask}, deterministic=deterministic)
        classify_loss = lasagne.objectives.categorical_crossentropy(pred, token) * weight
        classify_acc_1 = lasagne.objectives.categorical_accuracy(pred, token, top_k=1)
        classify_acc_5 = lasagne.objectives.categorical_accuracy(pred, token, top_k=5)

        return classify_loss, classify_acc_1, classify_acc_5

    def load_model(self, model_file):
        with open(model_file) as f:
            params_0 = pickle.load(f)

        lasagne.layers.set_all_param_values(self.net['classify'], params_0)
        glog.info('load model from %s' % os.path.basename(model_file))

    def load_model_feat(self, model_file):
        with open(model_file) as f:
            params_0 = pickle.load(f)
        params_0 = params_0[:6*2]	# params before fc6
        set_all_param_values(self.net['fc6'], params_0)
        glog.info('load feat model from %s' % os.path.basename(model_file))

    # def load_caffemodel(self, proto, caffemodel):
    # 	caffe.set_mode_cpu()
    # 	net = caffe.Net(proto, caffemodel, caffe.TEST)
    # 	for k, v in net.params.items():
    # 		if not k.startswith('fc'):
    # 			if k == 'conv1':
    # 				self.net[k].W.set_value(v[0].data[:, ::-1, :, :])
    # 			else:
    # 				self.net[k].W.set_value(v[0].data)
    # 			self.net[k].b.set_value(np.squeeze(v[1].data))
    # 			glog.info('layer [%s] loaded from caffemodel' % k)

    def save_model(self, model_file):
        params_0 = lasagne.layers.get_all_param_values(self.net['classify'])
        # params_1 = lasagne.layers.get_all_param_values(self.net['out_lin'])

        with open(model_file, 'wb') as f:
            pickle.dump(params_0, f)

    def set_learning_rate(self, to_lr=None):
        if not to_lr is None:
            self.learning_rate.set_value(to_lr)
            glog.info('Auto change learning rate to %.2e' % to_lr)
        else:
            self.config.load_config()
            if 'lr_change' in self.config.items.keys():
                lr = np.float32(self.config.items['lr_change'])
                if not lr == self.learning_rate:
                    self.learning_rate.set_value(lr)
                    glog.info('Change learning rate to %.2e' % lr)

    # def log_self(self):
    # 	filename = os.path.abspath(__file__)
    # 	if filename.endswith('c'):
    # 		filename=filename[:-1]
    # 	with open(filename) as f:
    # 		glog.info(f.read())
