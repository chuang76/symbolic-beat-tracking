import os 
import glob 
import numpy as np
import pandas as pd
import pickle 

sz = 1200

def main():

    directory = './input/raw/'
    if not os.path.exists(directory):
        os.makedirs(directory)  

    directory = './input/npz/'
    if not os.path.exists(directory):
        os.makedirs(directory) 

    # prepare the profiles
    PF = []   
    for i in range(21, 109):
        PF.append(str(i) + '/onset')
    for i in range(21, 109):
        PF.append(str(i) + '/dura')

    file_list = np.sort(glob.glob('./input/*.csv', recursive=False)).tolist()

    for file_idx in range(len(file_list)):
        data = pd.read_csv(file_list[file_idx])
        name = file_list[file_idx].split('/')[2]
        z = pd.DataFrame(np.zeros((int(data['end_time'].max() / 44100) * 100, 1)))
        new = pd.DataFrame()
        for i in range(len(PF)):
            new[PF[i]] = z[0]             
        new['IOI'], new['beat'], new['downbeat'] = z[0], z[0], z[0]   

        # caculate onset and duration 
        onset_arr = []                     
        for i in range(len(data)):
            pitch = data['note'][i]
            onset, offset = int(np.round(data['start_time'][i] / 44100 * 100, 0)), int(np.round(data['end_time'][i] / 44100 * 100, 0))
            onset_arr.append(onset)

            col = str(pitch) + '/onset'                   
            new[col][onset] = 1
            col = str(pitch) + '/dura'
            new[col][onset:offset+1] = 1 

        onset_arr = np.array(onset_arr)
        onset_arr = np.unique(onset_arr)
        onset_list = onset_arr.tolist()

        # caculate IOI 
        for i in range(len(onset_list)):
            index = onset_list[i]
            if i == 0:
                num = 0.0
                new['IOI'][index] = num
            else:
                num = np.round(float(onset_list[i] - onset_list[i-1]) * 0.01, 2)
                new['IOI'][index] = num

        # write to file 
        new.to_csv('./input/raw/' + str(name))

        X = pd.read_csv('./input/raw/' + str(name))
        Yb, Yd = pd.DataFrame(X, columns=['beat']).to_numpy(), pd.DataFrame(X, columns=['downbeat']).to_numpy()
        X = X.drop(['beat', 'downbeat', 'Unnamed: 0'], axis=1).to_numpy()   

        # calculate spectral flux (with onset feature)
        X_sf = np.zeros((len(X), 88))
        for i in range(len(X)):
            if i != len(X) - 1:
                X_sf[i] = np.maximum(X[i+1][:88] - X[i][:88], 0)
        X_sf = np.sum(X_sf, axis=1)
        X_sf = X_sf[:, np.newaxis]
        X = np.concatenate((X, X_sf), axis=1)       # concat 

        # tranform into frame-level data 
        N = int(len(X) / sz)
        if float(len(X) / sz) - int(len(X) / sz) == 0:
            need_pad = 0
        else:
            pad = X[N * sz:]
            need_pad = sz - len(pad)

        # prepare X_frames 
        X_tot = np.concatenate([X, np.zeros((need_pad, 178))])
        M = len(X_tot) / sz
        X_frames = []
        for idx in range(int(M)):
            X_frames.append(X_tot[idx * sz:(idx + 1) * sz])

        # prepare Y_frames 
        Y_need_pad = np.zeros((need_pad, 1))
        Yb_tot, Yd_tot = np.concatenate([Yb, Y_need_pad]), np.concatenate([Yd, Y_need_pad])
        Yb_frames, Yd_frames = [], []
        for idx in range(int(M)):
            Yb_frames.append(Yb_tot[idx * sz:(idx + 1) * sz])
            Yd_frames.append(Yd_tot[idx * sz:(idx + 1) * sz])
        Yb_frames, Yd_frames = np.squeeze(Yb_frames, axis=2), np.squeeze(Yd_frames, axis=2)

        # write to npz file
        name = name.split('.')[0] + '.npz'
        with open('./input/npz/' + str(name), 'wb') as f:
            pickle.dump([X_frames, Yb_frames, Yd_frames], f)


if __name__ == '__main__':
    main()