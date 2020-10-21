import os
import glob 
import numpy as np
import pickle
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sz = 600
seq_len = sz * 2 
input_dim, hidden_dim, beat_dim, downbeat_dim = 178, 25, seq_len * 2, seq_len * 2        
lr, epochs, batch_sz, n_weights, gamma = 0.01, 50, 8, 5, 0.1
folder = 'attn'

class custom_dataset(Dataset):
    def __init__(self, x, yb, yd):
        self.x = x 
        self.yb, self.yd = yb, yd
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.yb[idx]), torch.FloatTensor(self.yd[idx])

class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):

        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, beat_dim, downbeat_dim):
        super(Model, self).__init__()

        # setting 
        self.flag = True
        self.layer_sz = 2

        # layer 
        self.norm = nn.LayerNorm([seq_len, input_dim])
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=self.flag, num_layers=self.layer_sz)  
        for name, param in self.rnn.named_parameters():                   # initialization
            if 'bias' in name:
                nn.init.uniform_(param, a=-0.1, b=0.1)
            elif 'weight' in name:
                nn.init.uniform_(param, a=-0.1, b=0.1)

        self.beat = nn.ModuleList([nn.Linear(hidden_dim, 1), \
                                   nn.Linear(hidden_dim, 1)])          

        self.attn = Attention(hidden_dim)
        self.act = nn.Sigmoid()
    
    def forward(self, raw):      

        x = self.norm(raw)
        out, (hn, cn) = self.rnn(x) 
        out_tmp = torch.chunk(out, 2, -1)
        out_tmp = out_tmp[0] + out_tmp[1] 
        hn = hn.permute(1, 0, 2)
        attn_out, weights = self.attn(out_tmp, hn)     # (8, 1201, 25)

        # beat 
        b1, d1 = self.beat[0](attn_out), self.beat[1](attn_out)                   # (8, 1201, 1)
        b, d = self.act(b1), self.act(d1)
        b, d = b.squeeze(-1), d.squeeze(-1)                   # (8, 1201)

        return b, d 

def main():

    with open('./dataset/large_train_data.npz', 'rb') as f:
        data = pickle.load(f)
    X_data, Yb_data, Yd_data = data[0], data[1], data[2]
    dataset = custom_dataset(X_data, Yb_data, Yd_data)
    train_loader = DataLoader(dataset, batch_size=batch_sz, drop_last=True)
    print('train dataset done.')

    del dataset
    del X_data, Yb_data, Yd_data

    # model, optim, loss 
    model = Model(input_dim, hidden_dim, beat_dim, downbeat_dim)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)           
    beat_loss = nn.BCELoss()

    train_arr, test_arr, lr_arr = [], [], []

    for epoch in range(epochs):

        lr_data = optimizer.param_groups[0]['lr']
        # print('lr_data =', lr_data)

        print('\n[info] epoch %d' %(epoch), end='\t')

        model.train()                               
        bl_loss, dl_loss = 0, 0    

        for idx, (x, yb, yd) in enumerate(train_loader):

            x, yb, yd = x.to(device), yb.to(device), yd.to(device)
            b, d = model.forward(x)

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
        #     b, d = model.forward(x)
        #     bl, dl = beat_loss(b, yb), beat_loss(d, yd)
        #     bl_loss += bl.item()                
        #     dl_loss += dl.item()

        # print('[....] test loss = %.4f | beat loss = %.4f | downbeat loss = %.4f' % ((bl_loss + dl_loss) / idx, bl_loss / idx, dl_loss / idx))
        # test_arr.append([epoch, np.round(bl_loss / idx, 4), np.round(dl_loss / idx, 4)])

        scheduler.step()

        # lr_arr.append(lr_data)
        # with open('./loss/' + folder + '_loss.npz', 'wb') as f:
        #     pickle.dump([train_arr, test_arr, lr_arr], f)

        model_name = './models/' + folder + '_' + str(epoch) +'.pkl'
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_name)   

if __name__ == '__main__':
    main()