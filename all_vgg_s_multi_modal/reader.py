import os
import cv2
import sys
import glog
import h5py
import random
import pickle
import numpy as np

data_dir = '/var/disk1/RWTH2014'
database_file = os.path.join(data_dir, 'database_2014_combine.pkl')
data_file = os.path.join(data_dir, 'feat_multimodal.h5')
sys.path.insert(0, '/home/liuhu/workspace/journal/all_vgg_s_multi_modal')
from utils_multi import *

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

        h = h5py.File(data_file)

        self.image = [h['right_images'], h['left_images']]
        self.oflow = [h['right_of'], h['left_of']]
        self.coord = [h['right_coord'], h['left_coord'], h['head_coord']]

        self.batch_size = batch_size

        # PCA on RGB pixels for color shifts
        px = self.image[0][sorted(np.random.choice(799006, 300, replace=False))]
        px = np.concatenate((px, self.image[1][sorted(np.random.choice(799006, 300, replace=False))]), axis=0)
        px = px.reshape((-1, 3)) / 255.
        px -= px.mean(axis=0)
        self.eig_value, self.eig_vector = np.linalg.eig(np.cov(px.T))

        with open('/home/runpeng/workspace/coding_trial/process_RWTH/coord.pkl') as f:
            d = pickle.load(f)
            self.mean_coord = np.array(d['coord_mean'], dtype=np.float32)
            self.std_coord = np.array(d['coord_std'], dtype=np.float32)

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


    def iterate(self, return_folder=False, return_upsamp_indices=False):
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
            b_idx = [b for b in self.db['begin_index']]
            e_idx = self.db['end_index']

            x_len = [e_idx[i] - b_idx[i] for i in indices]

            if self.resample_at_end:    # for estimate
                h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride))) + 1 for l in x_len]
                X_Len = [(l - 1) * self.depth_stride + self.c3d_depth for l in h_len]
                upsamp_indices = upsampling_at_end(x_len, self.c3d_depth, self.depth_stride)
            elif self.resample == True: # for training
                X_Len, upsamp_indices = resampling(x_len, self.c3d_depth, self.depth_stride)
                h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride)))+1 for l in X_Len]
            else:                       # for test and final prediction
                h_len = [int(np.ceil(float(l - self.c3d_depth) / float(self.depth_stride)))+1 for l in x_len]
                X_Len = [(l-1)*self.depth_stride + self.c3d_depth for l in h_len]
                upsamp_indices = upsampling(x_len, self.c3d_depth, self.depth_stride)

            max_h_len = np.max(h_len)
            max_X_Len = np.max(X_Len)

            image = np.zeros((batch_size, max_X_Len, 2, 3, 101, 101), dtype=np.float32)
            oflow = np.zeros((batch_size, max_X_Len, 2, 2, 101, 101), dtype=np.float32)
            coord = np.zeros((batch_size, 20, max_X_Len), dtype=np.float32)
            mask = [np.concatenate((np.ones(l), np.zeros(max_h_len - l))) for l in h_len]

            # gathering features
            for i, ind in enumerate(indices):
                image_raw, warp_mat = self.get_imgs(b_idx[ind], e_idx[ind]) # (x_len, 2, 3, 101, 101)
                image_aug = interp_images(image_raw, upsamp_indices[i])  # (X_Len, 2, 3, 101, 101)	# interp_images means interp without float
                assert X_Len[i] == image_aug.shape[0]
                image[i, :X_Len[i]] = image_aug

                oflow_raw = self.get_oflow(b_idx[ind], e_idx[ind], warp_mat)
                oflow_aug = interp_images(oflow_raw, upsamp_indices[i])  # (X_Len, 2, 2, 101, 101)
                assert X_Len[i] == oflow_aug.shape[0]
                oflow[i, :X_Len[i]] = oflow_aug

                coord_raw = self.get_coord(range(b_idx[ind], e_idx[ind]))   # have problem due to position augmentation
                coord_aug = interp_images(coord_raw, upsamp_indices[i])  # (X_Len, 20)
                coord[i, :, :X_Len[i]] = coord_aug.transpose([1, 0])


            image = np.transpose(image, (2, 0, 1, 3, 4, 5))  # (2, batch_size, max_X_Len, 3, 101, 101)
            oflow = np.transpose(oflow, (2, 0, 1, 3, 4, 5))  # (2, batch_size, max_X_Len, 2, 101, 101)

            image = np.reshape(np.float32(image), (-1, 3, 101, 101))  # (2 * batch_size * X_len, 3, 101, 101)
            oflow = np.reshape(np.float32(oflow), (-1, 2, 101, 101)) / 20. * 128.  # (2 * batch_size * X_len, 2, 101, 101)
            coord = np.float32(coord)
            mask = np.array(mask, dtype=np.float32)  # (batch_size, max_h_len)

            yields = [image, oflow, coord, mask, token]
            if return_folder:
                yields += [ID]
            if return_upsamp_indices:
                yields += [upsamp_indices]

            yield yields

    def get_coord(self, indexs):
        # return features of trajectory
        feats = np.zeros((len(indexs), 20), dtype=np.float32)
        for i, index in enumerate(indexs):
            feats[i, 0: 2] = self.coord[0][index]
            feats[i, 2: 4] = self.coord[1][index]
            feats[i, 4: 6] = self.coord[0][index] - self.coord[2][index]
            feats[i, 6: 8] = self.coord[1][index] - self.coord[2][index]
            feats[i, 8:10] = self.coord[0][index] - self.coord[1][index]

        feats[:, 10: ] = diff_locations(feats[:, : 10])
        feats = (feats - self.mean_coord) / self.std_coord
        return feats


    def get_oflow(self, begin_index, end_index, mat):
        # return optical flow
        imgs = np.zeros((end_index-begin_index, 2, 101, 101, 2), dtype=np.float32)
        imgs_orig = np.zeros((end_index-begin_index, 2, 96, 96, 2), dtype=np.float32)
        for k in range(2):
            imgs_orig[:, k] = self.oflow[k][begin_index: end_index].astype(np.float32)
            for i in range(end_index-begin_index):
                imgs[i, k] = cv2.resize(imgs_orig[i, k], (101, 101))

        # for i, index in enumerate(indexs):
        #     for k in xrange(2):   # [right, left]
        #         img = self.oflow[k][index].astype(np.float32)
        #         img_resize = cv2.resize(img, (101, 101))
        #         imgs[i, k] = img_resize

        imgs = np.reshape(imgs, (-1, 101, 101, 2))
        imgs = of_augmentation(imgs, mat)
        imgs = np.transpose(np.reshape(imgs, (-1, 2, 101, 101, 2)), (0, 1, 4, 2, 3))  # (len, 2, 2, 101, 101)
        return imgs

    def get_imgs(self, begin_index, end_index):
        # same augmentation for one sentence
        # return RGB images
        imgs = np.zeros((end_index-begin_index, 2, 101, 101, 3), dtype=np.float32)
        imgs_orig = np.zeros((end_index-begin_index, 2, 96, 96, 3), dtype=np.float32)

        mean_file = np.array([123, 117, 102], dtype=np.float32)
        for k in xrange(len(self.image)):
            imgs_orig[:, k] = self.image[k][begin_index: end_index].astype(np.float32)
            for i in range(end_index-begin_index):
                imgs[i, k] = cv2.resize(imgs_orig[i, k] - mean_file[None, None, :], (101, 101))

        # for i, index in enumerate(indexs):
        #     for k in xrange(len(self.image)):
        #         img = self.image[k][index].astype(np.float32)
        #         img = img - mean_file[None, None, :]
        #         img_resize = cv2.resize(img, (101, 101))
        #
        #         imgs[i, k] = img_resize

        imgs = np.reshape(imgs, (-1, 101, 101, 3))
        imgs, mat = im_augmentation(imgs, self.eig_value, self.eig_vector, trans=0.1, color_dev=0.2, distortion=self.distortion)
        imgs = np.transpose(np.reshape(imgs, (-1, 2, 101, 101, 3)), (0, 1, 4, 2, 3))
        return imgs, mat


    def check_inputs(self, indices):
        batch_size = len(indices)

        y_len = np.array([len(self.db['token'][i]) for i in indices], dtype=np.int32)
        max_y_len = max(y_len)
        token = np.array([np.concatenate([self.tokens[idx], (max_y_len-y_len[i])*[-2]])+1 for i,idx in enumerate(indices)], dtype=np.int32)

        ID = [self.db['folder'][i] for i in indices]
        b_idx = [b for b in self.db['begin_index']]
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

        image = np.zeros((batch_size, max_X_Len, 2, 3, 101, 101), dtype=np.float32)
        oflow = np.zeros((batch_size, max_X_Len, 2, 2, 101, 101), dtype=np.float32)
        coord = np.zeros((batch_size, max_X_Len, 20), dtype=np.float32)
        mask = [np.concatenate((np.ones(l), np.zeros(max_h_len - l))) for l in h_len]

        # gathering features
        for i, ind in enumerate(indices):
            image_raw, warp_mat = self.get_imgs(range(b_idx[ind], e_idx[ind])) # (x_len, 2, 3, 101, 101)
            image_aug = interp_images(image_raw, upsamp_indices[i])  # (X_Len, 2, 3, 101, 101)  # interp_images means interp without float
            assert X_Len[i] == image_aug.shape[0]
            image[i, :X_Len[i]] = image_aug

            oflow_raw = self.get_oflow(range(b_idx[ind], e_idx[ind]), warp_mat)
            oflow_aug = interp_images(oflow_raw, upsamp_indices[i])  # (X_Len, 2, 2, 101, 101)
            assert X_Len[i] == oflow_aug.shape[0]
            oflow[i, :X_Len[i]] = oflow_aug

            coord_raw = self.get_coord(range(b_idx[ind], e_idx[ind]))
            coord_aug = interp_images(coord_raw, upsamp_indices[i])  # (X_Len, 20)
            coord[i, :X_Len[i]] = coord_aug

        image = np.transpose(image, (2, 0, 1, 3, 4, 5))  # (2, batch_size, max_X_Len, 3, 101, 101)
        oflow = np.transpose(oflow, (2, 0, 1, 3, 4, 5))  # (2, batch_size, max_X_Len, 2, 101, 101)

        image = np.reshape(np.float32(image), (-1, 3, 101, 101))  # (2 * batch_size * X_len, 3, 101, 101)
        oflow = np.reshape(np.float32(oflow), (-1, 2, 101, 101)) / 20. * 128.  # (2 * batch_size * X_len, 2, 101, 101)
        coord = np.float32(coord)
        mask = np.array(mask, dtype=np.float32)  # (batch_size, max_h_len)

        return image, oflow, coord, mask, token, ID, upsamp_indices


if __name__ == '__main__':
    train_set = Reader(phase='dev', batch_size=1, do_shuffle=False, resample=False, distortion=False)
    for inputs in train_set.iterate(return_folder=True):
        glog.info([s.shape for s in inputs[:-1]])
    exit(0)

    check_indices = np.array([  30,   32,   55])

    # image, oflow, coord, mask, token, ID, upsamp_indices = train_set.check_inputs(check_indices, offset=0)
    for image, oflow, coord, mask, token in train_set.iterate():
        break
    batch_size = mask.shape[0]
    image = np.transpose(np.reshape(image, (2, batch_size, -1, 3, 101, 101)), (1, 2, 0, 4, 5, 3))
    oflow = np.transpose(np.reshape(oflow, (2, batch_size, -1, 2, 101, 101)), (1, 2, 0, 4, 5, 3))

    import skimage.io as sio
    from skimage.transform import resize

    # os.system('mkdir check_imgs')

    for idx in xrange(train_set.batch_size):
        print idx, token[idx]
        print image.shape, oflow.shape
        print coord[idx][:, 0]
        print coord[idx][:, 4]
        print coord[idx][:, 10]

        img = image[idx] + np.array([123, 117, 102], dtype=np.float32)[None, None, None, None, :]
        ofl = oflow[idx]

        out = np.zeros((101 * 4, 101 * img.shape[0], 3), dtype=np.float32)
        img[img > 255.] = 255.
        img[img < 0.] = 0.
        
        for i in range(img.shape[0]):
            for j in range(2):
                im = img[i, j] / 255.
                out[j * 101: (j + 1) * 101, i * 101: (i + 1) * 101, :] = im

                of = flow2image_normal(ofl[i, j] / 128. * 20.)
                out[(j+2)*101: (j+3) * 101, i * 101: (i + 1) * 101, :] = of
        
        sio.imsave('reader.jpg', out)
        break


    # train_set = Reader(phase='train', batch_size=2, do_shuffle=True, resample=True, distortion=True)
    # for _ in range(10):
    #     image, oflow, coord, mask, token, ID, upsamp_indices = train_set.check_inputs(check_indices, offset=0)

    #     print 