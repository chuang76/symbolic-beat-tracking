import os
import glob 
import numpy as np
import pickle 

beat_thres, db_thres = 0.3, 0.2
file_list = np.sort(glob.glob('./tmp/attn/*.npz', recursive=True)).tolist() 

for file_idx in range(len(file_list)):

    name = file_list[file_idx].split('/')[3].split('.')[0]

    with open(file_list[file_idx], 'rb') as f: 
        data = pickle.load(f)

    b, d = data[0], data[1]
    b, d = np.squeeze(b), np.squeeze(d)
    b, d = b.tolist(), d.tolist()

    b_out, d_out = [], []
    for i in range(len(b)):
        if b[i] >= beat_thres:
            b_out.append(str(np.round(float(i) * 0.01, 2)) + '\n')
        if d[i] >= db_thres:
            d_out.append(str(np.round(float(i) * 0.01, 2)) + '\n')

    with open('./output/' + str(name) + '_beat_attn.txt', 'a') as fp:
        fp.writelines(b_out)

    with open('./output/' + str(name) + '_downbeat_attn.txt', 'a') as fp:
        fp.writelines(d_out)
