import os
import cv2
import sys
import glog
import h5py
import random
import pickle
import numpy as np
sys.path.insert(0, '/home/liuhu/workspace/journal/')
from utils import resampling, upsampling, upsampling_at_end, interp_images, im_augmentation

data_dir = '/var/disk1/RWTH2014'
database_file = os.path.join(data_dir, 'database_2014_combine.pkl')
data_file = '/var/disk1/RWTH2014/cropped_right_hand.h5'

class Reader(object):
    def __init__(self, phase, batch_size, c3d_depth=4, depth_stride=4, resample_at_end=False, resample=False, distortion=False, do_shuffle=False):
        # load database
        assert os.path.exists(database_file)
        # log_self(__file__)
        self.phase = phase
        with open(database_file) as f:
            db = pickle.load(f)
            self.db = db[phase]
            # {'folder': [], 'signer': [], 'annotation': [], 'vocabulary': [], 'token': [], 'begin_index': [], 'end_index': []}

        self.tokens = self.db['token']
        self.vocabulary = self.db['vocabulary']
        self.imgs = h5py.File(data_file)['images']
        self.batch_size = batch_size

        # PCA on RGB pixels for color shifts
        px = self.imgs[sorted(np.random.choice(799006, 1000, replace=False))]
        px = px.reshape((-1, 3)) / 255.
        px -= px.mean(axis=0)
        self.eig_value, self.eig_vector = np.linalg.eig(np.cov(px.T))

        self.resample_at_end = resample_at_end
        self.resample = resample
        self.distortion = distortion
        self.do_shuffle = do_shuffle
        self.c3d_depth = c3d_depth
        self.depth_stride = depth_stride
        self.vocabulary_size = len(self.db['vocabulary'])

        self.n_samples = len(self.db['folder'])
        h_len = [int(np.ceil(float(e - b - self.c3d_depth) / float(self.depth_stride))) + 1 for b, e in zip(self.db['begin_index'], self.db['end_index'])]
        X_len = [(l - 1) * self.depth_stride + self.c3d_depth for l in h_len]

        self.max_h_len = max(h_len)
        self.max_X_len = max(X_len)

    def iterate_imgs(self):
        # should only be used by cnn2d
        img_nums = self.imgs.shape[0]
        for k in range(0, img_nums, self.batch_size):
            batch_size = min(self.batch_size, img_nums - k)
            indices = range(k, k+batch_size)
            feat = self.get_imgs(indices)
            yield feat, k, k+batch_size

    def iterate(self, return_folder=False, offset=0, return_upsamp_indices=False):
        index = range(self.n_samples)
        if self.do_shuffle:
            random.shuffle(index)

        for k in range(0, self.n_samples, self.batch_size):
            indices = index[k: k+self.batch_size]
            batch_size = len(indices)

            y_len = np.array([len(self.db['token'][i]) for i in indices], dtype=np.int32)
            max_y_len = max(y_len)
            token = np.array([np.concatenate([self.tokens[idx], (max_y_len-y_len[i])*[-2]])+1 for i,idx in enumerate(indices)], dtype=np.int32)

            ID = [self.db['folder'][i] for i in indices]
            b_idx = [b + offset for b in self.db['begin_index']]
            e_idx = self.db['end_index']

            x_len = [e_idx[i] - b_idx[i] for i in indices]

            if self.resample_at_end:
                h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride))) + 1 for l in x_len]
                X_Len = [(l - 1) * self.depth_stride + self.c3d_depth for l in h_len]
                upsamp_indices = upsampling_at_end(x_len, self.c3d_depth, self.depth_stride)
            elif self.resample == True:
                X_Len, upsamp_indices = resampling(x_len, self.c3d_depth, self.depth_stride)
                h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride)))+1 for l in X_Len]
            else:
                h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride)))+1 for l in x_len]
                X_Len = [(l-1)*self.depth_stride + self.c3d_depth for l in h_len]
                upsamp_indices = upsampling(x_len, self.c3d_depth, self.depth_stride)

            max_h_len = np.max(h_len)
            max_X_Len = np.max(X_Len)

            feat = np.zeros((batch_size, max_X_Len, 3, 101, 101), dtype=np.float32)
            mask = [np.concatenate((np.ones(l), np.zeros(max_h_len - l))) for l in h_len]

            # gathering features
            for i, ind in enumerate(indices):
                feat_raw = self.get_imgs(range(b_idx[ind], e_idx[ind])) # (x_len, 3, 101, 101)
                feat_aug = interp_images(feat_raw, upsamp_indices[i])  	# (X_Len, 3, 101, 101)	# interp_images means interp without float
                assert X_Len[i] == feat_aug.shape[0]
                feat[i, :X_Len[i]] = feat_aug
                # feat[i] = np.concatenate((feat_aug, np.zeros((max_X_Len-X_Len[i], 3, 101, 101), dtype=np.float32)), axis=0)  # (max_X_Len, 1024)

            feat = np.reshape(np.float32(feat), (-1, 3, 101, 101))  # (batch_size*X_len, 3, 101, 101)
            mask = np.array(mask, dtype=np.float32)  # (batch_size, max_h_len)

            yields = [feat, mask, token]
            if return_folder:
                yields += [ID]
            if return_upsamp_indices:
                yields += [upsamp_indices]

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

    def check_inputs(self, indices, offset=0):
        batch_size = len(indices)

        y_len = np.array([len(self.db['token'][i]) for i in indices], dtype=np.int32)
        max_y_len = max(y_len)
        token = np.array([np.concatenate([self.tokens[idx], (max_y_len-y_len[i])*[-2]])+1 for i,idx in enumerate(indices)], dtype=np.int32)

        ID = [self.db['folder'][i] for i in indices]
        b_idx = [b + offset for b in self.db['begin_index']]
        e_idx = self.db['end_index']

        x_len = [e_idx[i] - b_idx[i] for i in indices]

        if self.resample_at_end:
            h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride))) + 1 for l in x_len]
            X_Len = [(l - 1) * self.depth_stride + self.c3d_depth for l in h_len]
            upsamp_indices = upsampling_at_end(x_len, self.c3d_depth, self.depth_stride)
        elif self.resample == True:
            X_Len, upsamp_indices = resampling(x_len, self.c3d_depth, self.depth_stride)
            h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride)))+1 for l in X_Len]
        else:
            h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride)))+1 for l in x_len]
            X_Len = [(l-1)*self.depth_stride + self.c3d_depth for l in h_len]
            upsamp_indices = upsampling(x_len, self.c3d_depth, self.depth_stride)

        max_h_len = np.max(h_len)
        max_X_Len = np.max(X_Len)

        feat = np.zeros((batch_size, max_X_Len, 3, 101, 101), dtype=np.float32)
        mask = [np.concatenate((np.ones(l), np.zeros(max_h_len - l))) for l in h_len]

        # gathering features
        for i, ind in enumerate(indices):
            feat_raw = self.get_imgs(range(b_idx[ind], e_idx[ind])) # (x_len, 3, 101, 101)
            feat_aug = interp_images(feat_raw, upsamp_indices[i])   # (X_Len, 3, 101, 101)  # interp_images means interp without float
            assert X_Len[i] == feat_aug.shape[0]
            feat[i, :X_Len[i]] = feat_aug
            # feat[i] = np.concatenate((feat_aug, np.zeros((max_X_Len-X_Len[i], 3, 101, 101), dtype=np.float32)), axis=0)  # (max_X_Len, 1024)

        feat = np.reshape(np.float32(feat), (-1, 3, 101, 101))  # (batch_size*X_len, 3, 101, 101)
        mask = np.array(mask, dtype=np.float32)  # (batch_size, max_h_len)

        yields = [feat, mask, token]
        yields += [ID]
        yields += [upsamp_indices]

        return yields

if __name__ == '__main__':
    train_set = Reader(phase='train', batch_size=2, do_shuffle=True, resample=True, distortion=True)
    check_indices = np.array([  30,   32,   55,  100,  101,  104,  198,  338,  770,  813, 1014,
        1049, 1066, 1312, 1314, 1485, 1486, 1565, 2010, 2133, 2390, 2571,
        3638, 3942, 3950, 3951, 3961, 4057, 4254, 4573, 4574, 5222, 5305,
        5342, 5402, 5645])

    feat, mask, token, ID, upsamp_indices = train_set.check_inputs(check_indices, offset=0)
    batch_size = mask.shape[0]
    feat = np.reshape(feat, (batch_size, -1, 3, 101, 101))

    import skimage.io as sio
    from skimage.transform import resize
    # os.system('mkdir check_imgs')
    for idx, indice in enumerate(check_indices):
        print idx, token[idx], ID[idx]
        img = np.transpose(feat[idx], (0, 2, 3, 1)) + np.array([123, 117, 102], dtype=np.float32)[None, None, None, :]
                
        out = np.zeros((101, 101 * img.shape[0], 3), dtype=np.float32)
        img[img > 255.] = 255.
        img[img < 0.] = 0.
        
        for i in range(img.shape[0]):
        	im = img[i] / 255.
        	out[:, i * 101: (i + 1) * 101, :] = im
        
        sio.imsave('check_imgs/%04d.jpg'%indice, out)
