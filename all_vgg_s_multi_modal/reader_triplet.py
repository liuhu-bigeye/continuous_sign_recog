import os
import sys
import h5py
import glog
import cv2
import pickle
import random
import numpy as np

from utils import *

data_dir = '/var/disk1/RWTH2014'
database_file = os.path.join(data_dir, 'database_2014_combine.pkl')
data_file = os.path.join(data_dir, 'feat_multimodal.h5')

class Reader(object):
    def __init__(self, phase, batch_size, distortion=False, seed=1234):
        random.seed(seed)

        self.phase = phase
        self.batch_size = batch_size
        self.distortion = False

        with open(database_file) as f:
            self.db = pickle.load(f)[phase]
            # ['vocabulary', 'begin_index', 'end_index', 'token', 'voca_len_mean', 'signer', 'folder', 'annotation']

        h = h5py.File(data_file)
        self.image = [h['right_images'], h['left_images']]
        self.oflow = [h['right_of'], h['left_of']]

        # PCA on RGB pixels for color shifts
        px = self.image[0][sorted(np.random.choice(799006, 300, replace=False))]
        px = np.concatenate((px, self.image[1][sorted(np.random.choice(799006, 300, replace=False))]), axis=0)
        px = px.reshape((-1, 3)) / 255.
        px -= px.mean(axis=0)
        self.eig_value, self.eig_vector = np.linalg.eig(np.cov(px.T))

        self.n_samples = len(self.db['token'])    # num of sentences
        self.data_flow = []
        self.shuffle()
        self.n_batch = len(self.data_flow)    # num of minibatches
        glog.info('dataset: %s, batch_size: %d, n_batch: %d'%(self.phase, self.batch_size, self.n_batch))

    def shuffle(self):
        self.data_flow = []
        index = range(self.n_samples)
        random.shuffle(index)
        for i in range(0, self.n_samples, self.batch_size):
            self.data_flow.append(index[i: i+self.batch_size])

    def iterate_minibatches(self, return_token=False, return_indexs=False):
        R = 3
        # choose one frame in each sentence as original
        # another frame from same sentense as positive
        # from another sentense(no same word) as false
        for indexs in self.data_flow:
            batch_size = len(indexs)
            # right, left, upper_body
            image = np.zeros((2, 3*batch_size, 3, 101, 101), dtype=np.float32)
            oflow = np.zeros((2, 3*batch_size, 2, 101, 101), dtype=np.float32)
            
            tri_indexs = []
            for i, index in enumerate(indexs):
                begin_index = self.db['begin_index'][index]
                end_index=self.db['end_index'][index]
                token=self.db['token'][index]

                # assign original
                chosen_index=random.choice(range(begin_index, end_index))
                image[:, i], warp_mat = self.get_imgs(chosen_index)
                oflow[:, i] = self.get_oflow(chosen_index, warp_mat)

                # assign positive
                positive_indexs=range(max(begin_index, chosen_index-R), chosen_index-1)+range(chosen_index+2, min(end_index, chosen_index+R+1))    #(-3 to -2, 2 to 3)
                if len(positive_indexs)==0:
                    positive_index=chosen_index
                else:
                    positive_index=random.choice(positive_indexs)

                stationary_flag = 0
                if chosen_index - begin_index < 0.35*(end_index - begin_index) or end_index - chosen_index < 0.35*(end_index - begin_index):
                    stationary_flag = 1

                image[:, i + batch_size], warp_mat = self.get_imgs(positive_index)
                oflow[:, i + batch_size] = self.get_oflow(positive_index, warp_mat)

                # assign negative
                negative_index=None
                while True:
                    negative_index=random.choice(range(self.n_samples))
                    if len(set(token)&set(self.db['token'][negative_index]))==0:
                        negative_token = self.db['token'][negative_index]

                        if stationary_flag == 0:
                            negative_index = random.choice(range(self.db['begin_index'][negative_index], self.db['end_index'][negative_index]))
                        else:  # ensure that the negative hand is not stationary
                            bidx, eidx = self.db['begin_index'][negative_index], self.db['end_index'][negative_index]
                            negative_index = random.choice(range(int(0.65*bidx + 0.35*eidx), int(0.35*bidx + 0.65*eidx)))
                        break

                image[:, i + batch_size*2], warp_mat = self.get_imgs(negative_index)
                oflow[:, i + batch_size*2] = self.get_oflow(negative_index, warp_mat)
                tri_indexs.append([chosen_index, positive_index, negative_index])

            mean_file = np.array([123, 117, 102], dtype=np.float32)
            image = np.float32(image - mean_file[None, None, :, None, None])
            oflow = np.float32(oflow) / 20. * 128.

            returns = [image, oflow]
            if return_token:
                returns.extend([token, negative_token])
            if return_indexs:
                returns.extend([tri_indexs])
            yield returns
        self.shuffle()


    def get_imgs(self, index):
        # return right hand resized
        img = np.zeros((2, 101, 101, 3), dtype=np.float32)
        for k in xrange(len(self.image)):
            im = self.image[k][index].astype(np.float32)
            im_resize = cv2.resize(im, (101, 101))

            img[k] = im_resize

        img, mat = im_augmentation(img, self.eig_value, self.eig_vector, trans=0.1, color_dev=0.2, distortion=self.distortion)
        img = np.transpose(img, (0, 3, 1, 2))  # (2, 3, 101, 101)

        return img, mat


    def get_oflow(self, index, mat):
        # return optical flow
        img = np.zeros((2, 101, 101, 2), dtype=np.float32)
        for k in xrange(len(self.image)):
            im = self.oflow[k][index].astype(np.float32)
            im_resize = cv2.resize(im, (101, 101))

            img[k] = im_resize

        img = of_augmentation(img, mat)
        img = np.transpose(img, (0, 3, 1, 2))  # (2, 2, 101, 101)

        return img


if __name__ == '__main__':
    img_save_dir = 'temp_img'
    train_set = Reader(phase='train', batch_size=6, distortion=True)
    mean_file = np.array([123, 117, 102], dtype=np.float32)
    for index, (image, oflow, token, negative_token, tri_indexs) in enumerate(train_set.iterate_minibatches(return_token=True, return_indexs=True)):
        print image.shape, oflow.shape
        batch_size = image.shape[1] / 3

        out = np.zeros((101 * 6, 101 * batch_size * 2, 3))
        
        image = np.reshape(np.transpose(image + mean_file[None, None, :, None, None], (0, 1, 3, 4, 2)), (2, 3, batch_size, 101, 101, 3))
        oflow = np.reshape(np.transpose(oflow / 128. * 20., (0, 1, 3, 4, 2)), (2, 3, batch_size, 101, 101, 2))
        for i in xrange(batch_size):
            for k in xrange(2):
                for m in xrange(3):
                    out[101 * m: 101 * (m + 1), 101 * (batch_size * k + i): 101 * (batch_size * k + i + 1), :] = image[k, m, i] / 255.
                    out[101*(m+3): 101*(m + 4), 101 * (batch_size * k + i): 101 * (batch_size * k + i + 1), :] = flow2image_normal(oflow[k, m, i])

        import skimage
        import skimage.io as sio
        sio.imsave('triplet.jpg', out)
        break