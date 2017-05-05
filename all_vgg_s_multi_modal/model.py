'''
                          _ooOoo_                                 
                         o8888888o                                    
                         88" . "88                                    
                         (| ^_^ |)                                    
                         O\  =  /O                                
                      ____/`---'\____                                                     
                    .'  \\|     |  `.                           
                   /  \\|||  :  |||  \                              
                  /  _||||| -:- |||||-  \                         
                  |   | \\\  -  / |   |                         
                  | \_|  ''\---/''  |   |                                 
                  \  .-\__  `-`  ___/-. /                                 
                ___`. .'  /--.--\  `. . ___                           
              ."" '<  `.___\_<|>_/___.'  >'"".                    
            | | :  `- \`.;`\ _ /`;.`/ - ` : | |                       
            \  \ `-.   \_ __\ /__ _/   .-` /  /                   
      ========`-.____`-.___\_____/___.-`____.-'========               
                           `=---='                                

no change, no bug

loss will converge quickly and smoothly

'''
import os
import glog
import pickle
import theano
import numpy as np
import theano.tensor as T
import lasagne

from lasagne.layers import *
from lasagne.nonlinearities import *
import caffe

import sys
sys.path.insert(0, '/home/liuhu/workspace/journal/try_ctc')
from m_ctc_cost import ctc_cost, best_right_path_cost, top_k_right_path_cost
from utils import Softmax

class Model(object):
    def __init__(self, phase, config, vocabulary_size=1295, hidden_ndim=512):
        # need to be same voca_size and hidde_ndim so as to load same shape params
        # self.log_self()
        size = 101
        # model paras
        self.config = config
        learning_rate = self.config.items['lr']
        self.alpha = np.array(1e-3, dtype=np.float32)
        self.eps = np.array(1e-6, dtype=np.float32)
        self.learning_rate = theano.shared(np.float32(config.items['lr']))

        self.nClasses = vocabulary_size + 1
        self.vocabulary_size = vocabulary_size

        # variables
        image = T.tensor4('image')  # (2*nb*len, 3, 101, 101) or (2*3*nb, 3, 101, 101)
        opflow = T.tensor4('opflow')  # (2*nb*len, 2, 101, 101)
        coord = T.tensor3('coord')  # (nb, len, 20)

        mask = T.matrix('mask')  # (nb, max_hlen)
        token = T.imatrix('token')  # (nb, max_vlen)
        # label = T.imatrix('label')  # (nb, voca_size)

        net = {}

        # RGB modal
        net['image'] = InputLayer(shape=(None, 3, size, size))  # (2*nb*len, 3, 101, 101)

        net['conv1'] = Conv2DLayer(incoming=net['image'], num_filters=96, filter_size=7, stride=2)
        net['norm1'] = LocalResponseNormalization2DLayer(incoming=net['conv1'])
        net['pool1'] = MaxPool2DLayer(incoming=net['norm1'], pool_size=3)

        net['conv2'] = Conv2DLayer(incoming=net['pool1'], num_filters=256, filter_size=5)
        net['pool2'] = MaxPool2DLayer(incoming=net['conv2'], pool_size=2)

        net['conv3'] = Conv2DLayer(incoming=net['pool2'], num_filters=512, filter_size=3, pad=1)
        net['conv4'] = Conv2DLayer(incoming=net['conv3'], num_filters=512, filter_size=3, pad=1)
        net['conv5'] = Conv2DLayer(incoming=net['conv4'], num_filters=512, filter_size=3, pad=1)

        
        # optical flow modal
        net['opflow'] = InputLayer(shape=(None, 2, size, size))  # (2*nb*len, 2, 101, 101)

        net['conv1_of'] = Conv2DLayer(incoming=net['opflow'], num_filters=96, filter_size=7, stride=2)
        net['norm1_of'] = LocalResponseNormalization2DLayer(incoming=net['conv1_of'])
        net['pool1_of'] = MaxPool2DLayer(incoming=net['norm1_of'], pool_size=3)

        net['conv2_of'] = Conv2DLayer(incoming=net['pool1_of'], num_filters=256, filter_size=5)
        net['pool2_of'] = MaxPool2DLayer(incoming=net['conv2_of'], pool_size=2)

        net['conv3_of'] = Conv2DLayer(incoming=net['pool2_of'], num_filters=512, filter_size=3, pad=1)
        net['conv4_of'] = Conv2DLayer(incoming=net['conv3_of'], num_filters=512, filter_size=3, pad=1)
        net['conv5_of'] = Conv2DLayer(incoming=net['conv4_of'], num_filters=512, filter_size=3, pad=1)


        # modal fusion 1
        net['fusion_1'] = ElemwiseSumLayer(incomings=[net['conv5'], net['conv5_of']])
        net['pool5'] = MaxPool2DLayer(incoming=net['fusion_1'], pool_size=3) # (2*nb*len, 512, 2, 2)

        net['fc6'] = DenseLayer(incoming=net['pool5'], num_units=1024)  # (2*nb*len, 1024) or (2*3*nb, 1024)
        net['fc6_trans'] = FlattenLayer(DimshuffleLayer(ReshapeLayer(incoming=net['fc6'], shape=(2, -1, 1024)), (1, 2, 0)), 
                                        outdim=2)  # (nb*len, 2048) or (3*nb, 2048)

        net['drop6'] = DropoutLayer(incoming=net['fc6_trans'], p=0.5)
        net['fc7'] = DenseLayer(incoming=net['drop6'], num_units=256, nonlinearity=identity)  # (3*nb, 256)


        # encoding network for image features
        net['mask'] = InputLayer(shape=(None, None), name='mask')  # (nb, max_hlen)
        self.nb = mask.shape[0]
        self.max_hlen = mask.shape[1]

        net['pre_conv1d'] = DimshuffleLayer(ReshapeLayer(incoming=net['fc6_trans'], shape=(self.nb, -1, 2048)), (0, 2, 1))
        net['conv1d_1'] = Conv1DLayer(net['pre_conv1d'], num_filters=1024, filter_size=3, pad='same')
        net['pool1d_1'] = MaxPool1DLayer(net['conv1d_1'], pool_size=2)    #(nb, 1024, max_hlen)
        net['drop1d_1'] = DropoutLayer(net['pool1d_1'], p=0.1, shared_axes=(2,))

        net['conv1d_2'] = Conv1DLayer(net['drop1d_1'], num_filters=1024, filter_size=3, pad='same')
        net['pool1d_2'] = MaxPool1DLayer(net['conv1d_2'], pool_size=2)    #(nb, 1024, max_hlen)
        net['drop1d_2'] = DropoutLayer(net['pool1d_2'], p=0.1, shared_axes=(2,))


        # location modal
        net['coord'] = InputLayer(shape=(None, 20, None))
        net['conv1d_1_cd'] = Conv1DLayer(net['coord'], num_filters=64, filter_size=3, pad='same')
        net['pool1d_1_cd'] = MaxPool1DLayer(net['conv1d_1_cd'], pool_size=2)  # (nb, 64, max_hlen)
        net['drop1d_1_cd'] = DropoutLayer(net['pool1d_1_cd'], p=0.1, shared_axes=(2,))

        net['conv1d_2_cd'] = Conv1DLayer(net['drop1d_1_cd'], num_filters=256, filter_size=3, pad='same')
        net['pool1d_2_cd'] = MaxPool1DLayer(net['conv1d_2_cd'], pool_size=2)  # (nb, 256, max_hlen)
        net['drop1d_2_cd'] = DropoutLayer(net['pool1d_2_cd'], p=0.1, shared_axes=(2,))


        # modal fusion 2
        net['fusion_2'] = ConcatLayer((net['drop1d_2'], net['drop1d_2_cd']), axis=1)  # (nb, 1280, max_hlen)


        # LSTM
        net['lstm_input'] = InputLayer(shape=(None, None, 1280), name='lstm_input')
        net['lstm_frw'] = LSTMLayer(incoming=net['lstm_input'], mask_input=net['mask'], forgetgate=Gate(b=lasagne.init.Constant(1.0)), num_units=hidden_ndim)  # (nb, max_hlen, hidden_ndim)
        net['lstm_bck'] = LSTMLayer(incoming=net['lstm_input'], mask_input=net['mask'], forgetgate=Gate(b=lasagne.init.Constant(1.0)), num_units=hidden_ndim, backwards=True)

        net['lstm_shp'] = ReshapeLayer(ConcatLayer((net['lstm_frw'], net['lstm_bck']), axis=2), shape=(-1, 2*hidden_ndim))  # (nb*max_hlen, 2*hidden_ndim)
        net['out'] = DenseLayer(net['lstm_shp'], self.nClasses, nonlinearity=identity)  # (nb*max_hlen, nClasses)
        net['out_lin'] = ReshapeLayer(net['out'], shape=(self.nb, -1, self.nClasses))


        self.net = net

        # try save load model
        dummy_save_file = 'dummy.pkl'
        glog.info('try save load dummy model to: %s...' % dummy_save_file)
        self.save_model(dummy_save_file)
        self.load_model(dummy_save_file)
        os.system('rm -rf dummy.pkl')
        glog.info('dummy save load success, remove it and start calculate outputs...')

        if phase == 'pretrain':
            # for triplet pretrain use
            self.params_feat = get_all_params(net['fc7'])
            regular_feat = lasagne.regularization.apply_penalty(self.params_feat, lasagne.regularization.l2) * np.array(5e-4 / 2, dtype=np.float32)
            
            ## triplet train loss
            triplet_loss_train = self.get_triplet_loss(image, opflow, deterministic=False)
            loss_train_feat = triplet_loss_train + regular_feat
            
            ## triplet valid loss
            triplet_loss_valid = self.get_triplet_loss(image, opflow, deterministic=True)
            loss_valid_feat = triplet_loss_valid + regular_feat
            
            self.updates = lasagne.updates.momentum(loss_train_feat, self.params_feat, learning_rate=learning_rate, momentum=0.9)
            self.inputs = [image, opflow]
            self.train_outputs = [loss_train_feat, triplet_loss_train]
            self.valid_outputs = [loss_valid_feat, triplet_loss_valid]

        elif phase == 'ctc':
            # for ctc loss
            self.params_full = lasagne.layers.get_all_params([self.net['fusion_2'], self.net['out_lin']], trainable=True)
            self.regular_params = lasagne.layers.get_all_params([self.net['fusion_2'], self.net['out_lin']], regularizable=True)
            regular_full = lasagne.regularization.apply_penalty(self.regular_params, lasagne.regularization.l2) * np.array(5e-4/2, dtype=np.float32)

            # full train loss
            ctc_loss_train, pred_train = self.get_ctc_loss(image, opflow, coord, mask, token, deteministic=False)
            loss_train_full = ctc_loss_train + regular_full

            # full valid loss
            ctc_loss_valid, pred_valid = self.get_ctc_loss(image, opflow, coord, mask, token, deteministic=True)
            loss_valid_full = ctc_loss_valid + regular_full

            self.updates = lasagne.updates.adam(loss_train_full, self.params_full, learning_rate=self.learning_rate)
            self.inputs = [image, opflow, coord, mask, token]
            self.train_outputs = [loss_train_full, ctc_loss_train, pred_train]
            self.valid_outputs = [loss_valid_full, ctc_loss_valid, pred_valid]

        elif phase == 'extract_feature':
            pass
            # # for feature extraction
            # fc6 = get_output(self.net['fc6'], data, deterministic = True)
            # self.feature_func = theano.function(inputs=[data], outputs=fc6)

        elif phase == 'get_prediction':
            embeding = get_output(self.net['fusion_2'], {self.net['image']: image, 
                                                         self.net['opflow']: opflow, 
                                                         self.net['coord']: coord}, deterministic=True)  # (nb, 1280, len_m)
            output_lin = get_output(self.net['out_lin'], {self.net['lstm_input']: T.transpose(embeding, (0, 2, 1)), self.net['mask']: mask}, deterministic=True)

            output_softmax = Softmax(output_lin)  # (nb, max_hlen, nClasses)
            output_trans = T.transpose(output_softmax, (1, 0, 2))  # (max_hlen, nb, nClasses)

            best_path_loss, best_path = best_right_path_cost(output_trans, mask, token)
            ctc_loss = ctc_cost(output_trans, T.sum(mask, axis=1, dtype='int32'), token)

            # (nb, max_hlen, voca_size+1)
            self.predict_func = theano.function(inputs=[data, mask, token], outputs=[best_path_loss, best_path, ctc_loss])

        elif phase == 'top_k_prediction':
            embeding = get_output(self.net['fusion_2'], {self.net['image']: image, 
                                                         self.net['opflow']: opflow, 
                                                         self.net['coord']: coord}, deterministic=True)  # (nb, 1280, len_m)
            output_lin = get_output(self.net['out_lin'], {self.net['lstm_input']: T.transpose(embeding, (0, 2, 1)), self.net['mask']: mask}, deterministic=True)

            output_softmax = Softmax(output_lin)  # (nb, max_hlen, nClasses)
            output_trans = T.transpose(output_softmax, (1, 0, 2))  # (max_hlen, nb, nClasses)

            top_k_path_loss, top_k_path = top_k_right_path_cost(output_trans, mask, token, k=config.items['top_k'])
            ctc_loss = ctc_cost(output_trans, T.sum(mask, axis=1, dtype='int32'), token)

            # (nb, max_hlen, voca_size+1)
            self.predict_func = theano.function(inputs=[data, mask, token], outputs=[output_lin, top_k_path_loss, top_k_path, ctc_loss])

        glog.info('Model built, phase = %s'%phase)


    def make_functions(self):
        self.train_func = theano.function(inputs=self.inputs, outputs=self.train_outputs, updates=self.updates)
        self.valid_func = theano.function(inputs=self.inputs, outputs=self.valid_outputs)


    def get_triplet_loss(self, image, opflow, deterministic=False):
        fc7 = get_output(self.net['fc7'], {self.net['image']: image, self.net['opflow']: opflow}, deterministic=deterministic)  # (3*nb, 256)
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


    def get_ctc_loss(self, image, opflow, coord, mask, token, deteministic=False):
        embeding = get_output(self.net['fusion_2'], {self.net['image']: image, 
                                                     self.net['opflow']: opflow, 
                                                     self.net['coord']: coord}, deterministic=deteministic)

        # loss calculation
        ## ctc loss
        output_lin = get_output(self.net['out_lin'], {self.net['lstm_input']: T.transpose(embeding, (0, 2, 1)), self.net['mask']: mask})
        output_softmax = Softmax(output_lin)  # (nb, max_hlen, nClasses)
        pred = T.argmax(output_softmax, axis=2)

        output_trans = T.transpose(output_softmax, (1, 0, 2))  # (max_hlen, nb, nClasses)
        ctc_loss = ctc_cost(output_trans, T.sum(mask, axis=1, dtype='int32'), token).mean()

        return ctc_loss, pred


    def load_model(self, model_file):
        with open(model_file) as f:
            params_0 = pickle.load(f)
            params_1 = pickle.load(f)

        lasagne.layers.set_all_param_values(self.net['fusion_2'], params_0)
        lasagne.layers.set_all_param_values(self.net['out_lin'], params_1)

        glog.info('load model from %s' % os.path.basename(model_file))


    def load_model_feat(self, model_file):
        pass
        # with open(model_file, 'rb') as f:
        #     params_0 = pickle.load(f)
        # params_0 = params_0[: 11 * 2]  # params before fc6
        # set_all_param_values(self.net['fc6'], params_0)
        # glog.info('load feat model from %s' % os.path.basename(model_file))


    # def load_model_before_lstm(self, model_file):
    #     with open(model_file, 'rb') as f:
    #         params_0 = pickle.load(f)
    #     params_0 = params_0[:8*2]   # params before fc6
    #     set_all_param_values(self.net['drop1d_2'], params_0)
    #     glog.info('load before lstm model from %s' % os.path.basename(model_file))


    def load_caffemodel(self, proto, caffemodel):
        caffe.set_mode_cpu()
        net = caffe.Net(proto, caffemodel, caffe.TEST)

        for k in ['conv%d'%i for i in range(1, 6)]:
            # for W: (out_channel, in_channel, filter_size, filter_size)
            W, b = net.params[k]
            if k == 'conv1':
                self.net[k].W.set_value(W.data[:, ::-1, :, :])
                self.net['%s_of' % k].W.set_value(W.data[:, 1::-1, :, :]*1.5)
            else:
                self.net[k].W.set_value(W.data)
                self.net['%s_of' % k].W.set_value(W.data)
            self.net[k].b.set_value(np.squeeze(b.data))
            self.net['%s_of' % k].b.set_value(np.squeeze(b.data))

            glog.info('layer [%s] loaded from caffemodel' % k)
            glog.info('layer [%s_of] loaded from caffemodel' % k)

        del net


    def save_model(self, model_file):
        params_0 = lasagne.layers.get_all_param_values(self.net['fusion_2'])
        params_1 = lasagne.layers.get_all_param_values(self.net['out_lin'])
        # params_2 = lasagne.layers.get_all_param_values(self.net['det_lin'])

        with open(model_file, 'wb') as f:
            pickle.dump(params_0, f)
            pickle.dump(params_1, f)
            # pickle.dump(params_2, f)

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
    #     filename = os.path.abspath(__file__)
    #     if filename.endswith('c'):
    #         filename=filename[:-1]
    #     with open(filename) as f:
    #         glog.info(f.read())
