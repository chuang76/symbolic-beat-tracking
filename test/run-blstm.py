import os
import glob 
import numpy as np
import pickle
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sz = 600
seq_len, batch_sz = sz * 2, 8
input_dim, hidden_dim, beat_dim, downbeat_dim = 178, 25, seq_len * 2, seq_len * 2

class custom_dataset(Dataset):
    def __init__(self, x, yb, yd):
        self.x = x 
        self.yb, self.yd = yb, yd
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.yb[idx]), torch.FloatTensor(self.yd[idx])

class Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, beat_dim, downbeat_dim):
        super(Model, self).__init__()

        # setting 
        self.flag = True
        self.layer_sz = 2
        self.bi_num = 2
        self.norm = nn.LayerNorm([seq_len, input_dim])
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=self.flag, num_layers=self.layer_sz)  

        # initialization 
        for name, param in self.rnn.named_parameters():                   
            if 'bias' in name:
                nn.init.uniform_(param, a=-0.1, b=0.1)
            elif 'weight' in name:
                nn.init.uniform_(param, a=-0.1, b=0.1)

        self.beat = nn.ModuleList([nn.Linear(hidden_dim * self.bi_num, 1), \
                                   nn.Linear(hidden_dim * self.bi_num, 1)])         
        self.act = nn.Sigmoid()

    def forward(self, x):      

        x = self.norm(x)
        out, (hn, cn) = self.rnn(x)
        b, d = self.beat[0](out), self.beat[1](out)                 
        b, d = self.act(b), self.act(d)
        b, d = b.squeeze(-1), d.squeeze(-1)               

        return b, d 

def main():

    print('Caculate BLSTM results.')
    
    file_list = np.sort(glob.glob('./input/npz/*.npz', recursive=True)).tolist() 
    model = Model(input_dim, hidden_dim, beat_dim, downbeat_dim)
    model = model.to(device)
    checkpoint = torch.load('./models/BLSTM.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for file_idx in range(len(file_list)):

        # create a folder to store frames
        name = file_list[file_idx].split('/')[3].split('.')[0]
        directory = './tmp/blstm/' + str(name)
        if not os.path.exists(directory):
            os.makedirs(directory)     

        # prepare dataset and loader 
        with open(file_list[file_idx], 'rb') as f:                      
            data = pickle.load(f)
        X_data, Yb_data, Yd_data = data[0], data[1], data[2]
        dataset = custom_dataset(X_data, Yb_data, Yd_data)
        loader = DataLoader(dataset, batch_size=batch_sz, drop_last=False)

        model.eval()  
        for idx, (x, yb, yd) in enumerate(loader):
            x, yb, yd = x.to(device), yb.to(device), yd.to(device)
            b, d = model.forward(x)
            b, d = b.cpu().detach().numpy(), d.cpu().detach().numpy()

            with open('./tmp/blstm/' + str(name) + '/' + str(idx) + '_unit.npz', 'wb') as fp:
                pickle.dump([b, d], fp)

        # output 
        tmp_list = np.sort(glob.glob('./tmp/blstm/' + str(name) + '/*.npz', recursive=True)).tolist() 
        b_arr, d_arr = [], []
        for idx in range(len(tmp_list)):
            with open(tmp_list[idx], 'rb') as f:
                data = pickle.load(f)
            b_arr.append(data[0])
            d_arr.append(data[1])

        beat, downbeat = np.concatenate(b_arr), np.concatenate(d_arr)
        beat, downbeat = beat.reshape(-1, 1), downbeat.reshape(-1, 1)

        with open('./tmp/blstm/' + str(name) + '.npz', 'wb') as f:
            pickle.dump([beat, downbeat], f) 


if __name__ == '__main__':
    main()