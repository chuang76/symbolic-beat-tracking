import os
import glob
import numpy as np
import pickle

data_list = np.sort(glob.glob('./dataset/npz_train/*.npz')).tolist()
# print('data_list len =', len(data_list))

X_data, Yb_data, Yd_data, Yt_data = [], [], [], []

for idx in range(len(data_list)):
    with open(data_list[idx], 'rb') as f:
        data = pickle.load(f)
        X_data.append(data[0])
        Yb_data.append(data[1])
        Yd_data.append(data[2])

X, Yb, Yd = np.concatenate(X_data), np.concatenate(Yb_data), np.concatenate(Yd_data)

with open('./dataset/large_train_data.npz', 'wb') as f:
    pickle.dump([X, Yb, Yd], f, protocol=4)
