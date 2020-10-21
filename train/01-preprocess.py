import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import bisect 
from scipy.interpolate import interp1d
import copy
import glob 

file_list = np.sort(glob.glob('./MusicNet/*.csv', recursive=True)).tolist()

directory = './dataset/training_dataset/'
    if not os.path.exists(directory):
        os.makedirs(directory)

directory = './dataset/npz_train/'
    if not os.path.exists(directory):
        os.makedirs(directory)

N = []   
for i in range(21, 109):
    N.append(str(i) + '/onset')
for i in range(21, 109):
    N.append(str(i) + '/dura')

for file_idx in range(len(file_list)):

    data = pd.read_csv(file_list[file_idx])
    name = file_list[file_idx].split('/')[2]

    z = np.zeros((int(data['end_time'].max() / 44100) * 100, 1))
    z = pd.DataFrame(z)
    new = pd.DataFrame()
    for i in range(len(N)):
        new[N[i]] = z[0]             
    new['IOI'], new['beat'], new['downbeat'] = z[0], z[0], z[0]   

    m = int(data['meter'][0].split('/')[0])                # meter 
    B = int(data['start_beat'].max())                      # to generate reference list 
    ref_list = np.arange(B + 1)   

    x_list, y_list = [], []
    for i in range(len(data)):
        if data['start_beat'][i].is_integer():
            y_list.append(data['start_beat'][i])
            x_list.append(np.around(data['start_time'][i] / 44100 * 100, 0))

    lost_list = []
    for i in range(len(ref_list)):
        if ref_list[i] not in y_list and ref_list[i] >= data['start_beat'].min():
            lost_list.append(ref_list[i])

    # interpolation
    f = interp1d(y_list, x_list, kind='linear', fill_value='extrapolate')
    insert_beat = []
    for i in range(len(lost_list)):
        idx = lost_list[i]
        t = np.round(f(idx), 0)  
        b = idx % m               
        if b == 1:
            d = 1
        else:
            d = 0 
        insert_beat.append([t, 1, d])

    # onset and duration 
    onset_arr = []                     
    for i in range(len(data)):
        pitch = data['note'][i]
        onset, offset = int(np.round(data['start_time'][i] / 44100 * 100, 0)), nt(np.round(data['end_time'][i] / 44100 * 100, 0))
        beat = data['start_beat'][i] 
        onset_arr.append(onset)
        
        new[str(pitch) + '/onset'][onset] = 1
        new[str(pitch) + '/dura'][onset:offset+1] = 1 
        
        if beat.is_integer():
            beat = beat % int(m)
        else:
            beat = -1                   # not beat 

        if beat != -1 and beat % int(m) == 1:
            new['downbeat'][onset] = 1
        
        if beat != -1:
            new['beat'][onset] = 1 

    onset_arr = np.array(onset_arr)
    onset_arr = np.unique(onset_arr)
    onset_list = onset_arr.tolist()

    # IOI 
    for i in range(len(onset_list)):
        if i == 0:
            num = 0.0
            new['IOI'][onset_list[i]] = num
        else:
            num = np.round(float(onset_list[i] - onset_list[i-1]) * 0.01, 2)
            new['IOI'][onset_list[i]] = num

    for i in range(len(insert_beat)):
        new['beat'][insert_beat[i][0]] = insert_beat[i][1]
        new['downbeat'][insert_beat[i][0]] = insert_beat[i][2]

    path = './dataset/training_dataset/' + name
    new.to_csv(path)

    # =========================================================================================

    X = pd.read_csv(path)
    name = path.split('/')[3].split('.')[0]
    Yb, Yd = pd.DataFrame(X, columns=['beat']).to_numpy(), pd.DataFrame(X, columns=['downbeat']).to_numpy()
    X = X.drop(['beat', 'downbeat', 'Unnamed: 0'], axis=1).to_numpy()   

    # setting 
    rows = len(X)
    z = np.zeros((rows, 1))
    z = pd.DataFrame(z)

    # spectral flux 
    X_sf = np.zeros((len(X), 88))
    for i in range(len(X)):
        if i != len(X) - 1:
            X_sf[i] = np.maximum(X[i+1][:88] - X[i][:88], 0)

    X_sf = np.sum(X_sf, axis=1)
    X_sf = X_sf[:, np.newaxis]
    X = np.concatenate((X, X_sf), axis=1)       # concat 

    num_frag = int(len(X) / sz)
    if num_frag * sz + sz > len(X):
        num_frag = num_frag - 1

    idx_list = []
    for i in range(num_frag):
        start_idx = sz * i
        end_idx = start_idx + (sz * 2)
        idx = int((start_idx + end_idx) / 2)
        idx_list.append(idx)

    # transform to frame-level 
    X_data = []
    for i in idx_list:    
        X_data.append(X[i-sz:i+sz])

    Yb_data, Yd_data = [], []
    for i in idx_list:    
        Yb_data.append(Yb[i-sz:i+sz])
        Yd_data.append(Yd[i-sz:i+sz])
    Yb_data, Yd_data = np.squeeze(Yb_data, axis=2), np.squeeze(Yd_data, axis=2)

    with open('./dataset/npz_train/' + name + '.npz', 'wb') as f:
        pickle.dump([X_data, Yb_data, Yd_data], f)

