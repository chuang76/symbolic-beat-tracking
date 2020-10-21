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
seq_len = sz * 2 
input_dim, hidden_dim, beat_dim, downbeat_dim = 178, 25, seq_len * 2, seq_len * 2        
lr, epochs, batch_sz, n_weights, gamma = 0.01, 50, 8, 5, 0.1
folder = 'blstm'

class custom_dataset(Dataset):
    def __init__(self, x, yb, yd):
        self.x = x 
        self.yb, self.yd = yb, yd
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.yb[idx]), torch.FloatTensor(self.yd[idx])

class Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(Model, self).__init__()

        # setting 
        self.flag = True
        self.layer_sz = 2
        self.bi_num = 2

        # layer 
        self.norm = nn.LayerNorm([seq_len, input_dim])
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=self.flag, num_layers=self.layer_sz)  
        for name, param in self.rnn.named_parameters():                   # initialization
            if 'bias' in name:
                nn.init.uniform_(param, a=-0.1, b=0.1)
            elif 'weight' in name:
                nn.init.uniform_(param, a=-0.1, b=0.1)

        self.beat = nn.ModuleList([nn.Linear(hidden_dim * self.bi_num, 1), \
                                   nn.Linear(hidden_dim * self.bi_num, 1)])         
        self.act = nn.Sigmoid()

    def forward(self, raw):     

        x = self.norm(raw)
        out, (hn, cn) = self.rnn(x)
        b, d = self.beat[0](out), self.beat[1](out)                   # (8, 1201, 1)
        b, d = self.act(b), self.act(d)
        b, d = b.squeeze(-1), d.squeeze(-1)                           # (8, 1201)
        return b, d, out    

def main():

    with open('./dataset/large_train_data.npz', 'rb') as f:
        data = pickle.load(f)
    X_data, Yb_data, Yd_data = data[0], data[1], data[2]
    dataset = custom_dataset(X_data, Yb_data, Yd_data)
    train_loader = DataLoader(dataset, batch_size=batch_sz, drop_last=True)

    # with open('./dataset/large_valid_data.npz', 'rb') as f:
    #     data = pickle.load(f)
    # X_data, Yb_data, Yd_data = data[0], data[1], data[2]
    # dataset = custom_dataset(X_data, Yb_data, Yd_data)
    # test_loader = DataLoader(dataset, batch_size=batch_sz, drop_last=True)

    del dataset
    del X_data, Yb_data, Yd_data

    model = Model(input_dim, hidden_dim)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)           
    beat_loss = nn.BCELoss()

    train_arr, test_arr = [], []

    for epoch in range(epochs):

        print('\n[info] epoch %d' %(epoch), end='\t')
        epoch = "%02d" %(epoch)

        lr_data = optimizer.param_groups[0]['lr']
        # print('lr_data =', lr_data)

        model.train()                             
        bl_loss, dl_loss = 0, 0      
        for idx, (x, yb, yd) in enumerate(train_loader):

            x, yb, yd = x.to(device), yb.to(device), yd.to(device)
            b, d, out = model.forward(x)
            bl, dl = beat_loss(b, yb), beat_loss(d, yd)
            bl_loss += bl.item()                
            dl_loss += dl.item()
            train_loss = bl + dl * n_weights

            optimizer.zero_grad()                # update 
            train_loss.backward()
            optimizer.step()

        print('train loss = %.4f | beat loss = %.4f | downbeat loss = %.4f' % ((bl_loss + dl_loss) / idx, bl_loss / idx, dl_loss / idx))
        train_arr.append([epoch, np.round(bl_loss / idx, 4), np.round(dl_loss / idx, 4)])

        # model.eval()
        # bl_loss, dl_loss = 0, 0
        # for idx, (x, yb, yd) in enumerate(test_loader):
        #     x, yb, yd = x.to(device), yb.to(device), yd.to(device)
        #     b, d, out = model.forward(x)
        #     bl, dl = beat_loss(b, yb), beat_loss(d, yd)
        #     bl_loss += bl.item()                
        #     dl_loss += dl.item()

        # print('test loss = %.4f | beat loss = %.4f | downbeat loss = %.4f' % ((bl_loss + dl_loss) / idx, bl_loss / idx, dl_loss / idx))
        # test_arr.append([epoch, np.round(bl_loss / idx, 4), np.round(dl_loss / idx, 4)])

        scheduler.step()

        # with open('./loss/' + folder + '_loss.npz', 'wb') as f:
        #     pickle.dump([train_arr, test_arr], f)

        model_name = './models/' + folder + '_' + str(epoch) +'.pkl'
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_name)   

if __name__ == '__main__':
    main()