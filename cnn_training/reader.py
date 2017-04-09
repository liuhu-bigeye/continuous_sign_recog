import os
import cv2
import sys
import glog
import h5py
import random
import pickle
import numpy as np
sys.path.insert(0, '/home/liuhu/workspace/journal/')
from utils import im_augmentation

# database_file = '/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_2017-03-28_20-41-05/ctc_best_path_63_0.412_0.411_off0.pkl'   # token, weight, indices, mask
database_file = '/home/liuhu/workspace/journal/all_vgg_s/output/ctc_predict_top_k_2017-04-05_20-02-09/ctc_best_path_63_0.412_0.411_top_10.pkl'   # token, weight, indices, mask
data_file = '/var/disk1/RWTH2014/cropped_right_hand.h5'

class Reader(object):
    def __init__(self, phase, batch_size, distortion=False, do_shuffle=False):
        # load database
        assert os.path.exists(database_file)
        # log_self(__file__)
        self.phase = phase
        with open(database_file) as f:
            db = pickle.load(f)
            self.db = db[phase]
            # {'folder': [], 'signer': [], 'annotation': [], 'vocabulary': [], 'token': [], 'begin_index': [], 'end_index': []}
        glog.info('dataset loaded...')

        self.tokens = self.db['token']
        self.weights = self.db['weight']
        self.indices = self.db['indices']
        self.masks = self.db['mask']

        self.imgs = h5py.File(data_file)['images']
        self.batch_size = batch_size

        # PCA on RGB pixels for color shifts
        px = self.imgs[sorted(np.random.choice(799006, 1000, replace=False))]
        px = px.reshape((-1, 3)) / 255.
        px -= px.mean(axis=0)
        self.eig_value, self.eig_vector = np.linalg.eig(np.cov(px.T))

        self.distortion = distortion
        self.do_shuffle = do_shuffle
        self.vocabulary_size = 1295

        self.n_samples = len(self.tokens)

    def iterate(self):
        index = range(self.n_samples)
        if self.do_shuffle:
            random.shuffle(index)

        for k in range(0, self.n_samples, self.batch_size):
            indices = index[k: k+self.batch_size]
            batch_size = len(indices)

            img_indices = [self.indices[i] for i in indices]
            x_len = map(len, img_indices)
            max_x_len = max(x_len)
            assert max_x_len<=10

            mask = [self.masks[i] for i in indices]
            assert x_len == map(sum, mask)
            mask_pre = map(lambda x:x.index(1), mask)

            token = np.array([self.tokens[i] for i in indices], dtype=np.int32)
            feat = np.zeros((batch_size, 10, 3, 101, 101), dtype=np.float32)
            weight = np.array([self.weights[i] for i in indices], dtype=np.float32)

            # gathering features
            for i in range(batch_size):
                feat_aug = self.get_imgs(img_indices[i])
                assert x_len[i] == feat_aug.shape[0]
                feat[i, mask_pre[i]:mask_pre[i]+x_len[i]] = feat_aug

            feat = np.reshape(np.float32(feat), (-1, 3, 101, 101))  # (batch_size*X_len, 3, 101, 101)
            mask = np.tile(np.array(mask, dtype=np.float32)[:, None, 1:-1:2], (1, 1024, 1))

            yields = [feat, token, weight, mask]

            yield yields

    def get_imgs(self, indexs):
        # return right hand resized
        imgs = np.zeros((len(indexs), 101, 101, 3), dtype=np.float32)
        mean_file = np.array([123, 117, 102], dtype=np.float32)
        for i, index in enumerate(indexs):
            img = self.imgs[index].astype(np.float32)
            img = img - mean_file[None, None, :]
            img_resized = cv2.resize(img, (101, 101))
            imgs[i] = img_resized
        im_augmentation(imgs, self.eig_value, self.eig_vector, trans=0.1, color_dev=0.2, distortion=self.distortion)
        return np.transpose(imgs, [0, 3, 1, 2])

if __name__ == '__main__':
    train_set = Reader(phase='train', batch_size=2, do_shuffle=True, distortion=True)
    for index, (feat, mask, token) in enumerate(train_set.iterate()):
        glog.info((index*2./5600, feat.shape, mask.shape, token.shape))
