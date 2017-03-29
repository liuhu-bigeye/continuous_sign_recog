import h5py
import sys
import numpy as np

if __name__ == '__main__':
    df_path = sys.argv[1]
    df_modified_path = df_path[:-3]+'_modified.h5'

    df_orig = h5py.File(df_path)
    df = h5py.File(df_modified_path, 'w')

    for key in df_orig.keys():
        if key.startswith('alignment'):
            df.create_dataset(key, shape=df_orig[key].shape, dtype=np.int32)
        else:
            df.create_dataset(key, shape=df_orig[key].shape, dtype=np.float32)
        df[key][...] = df_orig[key][...]

    n_samples = df['alignment_0'].shape[0]
    pred = [df['alignment_%d'%j][i, :int(sum(df['mask_%d'%j][i]))] for i in range(n_samples) for j in range(4)]
    non_zero_rate = sum(map(lambda x:sum(x!=0), pred)) / 1.0 / sum(map(len, pred))
    print 'non zero rate = %f'%non_zero_rate

    df.close()
    df_orig.close()