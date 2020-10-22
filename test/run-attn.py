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
        attn_out, weights = self.attn(out_tmp, hn)    

        # beat 
        b, d = self.beat[0](attn_out), self.beat[1](attn_out) 
        b, d = self.act(b), self.act(d)
        b, d = b.squeeze(-1), d.squeeze(-1)                   

        return b, d, out, attn_out   

def main():

    print('Calculate tracking results of BLSTM-Attn.')

    file_list = np.sort(glob.glob('./input/npz/*.npz', recursive=True)).tolist() 
    model = Model(input_dim, hidden_dim, beat_dim, downbeat_dim)
    model = model.to(device)
    checkpoint = torch.load('./models/BLSTM-Attn.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    s = './output/tmp/attn/'
    
    for file_idx in range(len(file_list)):

        # create a folder to store frames
        name = file_list[file_idx].split('/')[3].split('.')[0]
        directory = s + str(name)
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
            b, d, _, _ = model.forward(x)
            b, d = b.cpu().detach().numpy(), d.cpu().detach().numpy()

            with open(s + str(name) + '/' + str(idx) + '_unit.npz', 'wb') as fp:
                pickle.dump([b, d], fp)

        # output 
        tmp_list = np.sort(glob.glob(s + str(name) + '/*.npz', recursive=True)).tolist() 
        b_arr, d_arr = [], []
        for idx in range(len(tmp_list)):
            with open(tmp_list[idx], 'rb') as f:
                data = pickle.load(f)
            b_arr.append(data[0])
            d_arr.append(data[1])

        beat, downbeat = np.concatenate(b_arr), np.concatenate(d_arr)
        beat, downbeat = beat.reshape(-1, 1), downbeat.reshape(-1, 1)

        with open(s + str(name) + '.npz', 'wb') as f:
            pickle.dump([beat, downbeat], f) 

if __name__ == '__main__':
    main()