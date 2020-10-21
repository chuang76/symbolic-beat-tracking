import numpy as np
import pickle 
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn import Module, Parameter
from torch.autograd import Function, Variable
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seq_len, input_size, hidden_size = 600 * 2, 178, 25
batch_sz, batch_size, epochs, lr, n_weights = 64, 64, 50, 0.01, 5 
size_list = [hidden_size, hidden_size]
folder = 'hrnn'

class custom_dataset(Dataset):
    def __init__(self, x, yb, yd):
        self.x = x 
        self.yb, self.yd = yb, yd
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.yb[idx]), torch.FloatTensor(self.yd[idx])

def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output

class bound(Function):
    def forward(self, x):
        self.save_for_backward(x)
        output = x > 0.5
        return output.float()

    def backward(self, output_grad):
        x = self.saved_tensors
        x_grad = None

        if self.needs_input_grad[0]:
            x_grad = output_grad.clone()

        return x_grad

def repackage_hidden(h):
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class HM_LSTMCell(Module):
    def __init__(self, bottom_size, hidden_size, top_size, a, last_layer):
        super(HM_LSTMCell, self).__init__()
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.a = a
        self.last_layer = last_layer

        self.U_11 = Parameter(torch.cuda.FloatTensor(4 * self.hidden_size + 1, self.hidden_size))
        if not self.last_layer:
            self.U_21 = Parameter(torch.cuda.FloatTensor(4 * self.hidden_size + 1, self.top_size))
        self.W_01 = Parameter(torch.cuda.FloatTensor(4 * self.hidden_size + 1, self.bottom_size))
        self.bias = Parameter(torch.cuda.FloatTensor(4 * self.hidden_size + 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(self, c, h_bottom, h, h_top, z, z_bottom):
        s_recur = torch.mm(self.W_01, h_bottom)
        if not self.last_layer:
            s_topdown_ = torch.mm(self.U_21, h_top)
            s_topdown = z.expand_as(s_topdown_) * s_topdown_
        else:
            s_topdown = Variable(torch.zeros(s_recur.size()).cuda(), requires_grad=False).cuda()
        s_bottomup_ = torch.mm(self.U_11, h)
        s_bottomup = z_bottom.expand_as(s_bottomup_) * s_bottomup_

        f_s = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(1).expand_as(s_recur)
        # f_s.size = (4 * hidden_size + 1) * batch_size
        f = Func.sigmoid(f_s[0:self.hidden_size, :])  # hidden_size * batch_size
        i = Func.sigmoid(f_s[self.hidden_size:self.hidden_size*2, :])
        o = Func.sigmoid(f_s[self.hidden_size*2:self.hidden_size*3, :])
        g = Func.tanh(f_s[self.hidden_size*3:self.hidden_size*4, :])
        z_hat = hard_sigm(self.a, f_s[self.hidden_size*4:self.hidden_size*4+1, :])

        one = Variable(torch.ones(f.size()).cuda(), requires_grad=False)
        z = z.expand_as(f)
        z_bottom = z_bottom.expand_as(f)

        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = z * o * Func.tanh(c_new) + (one - z) * (one - z_bottom) * h + (one - z) * z_bottom * o * Func.tanh(c_new)

        z_new = bound()(z_hat)

        return h_new, c_new, z_new

class HM_LSTM(Module):
    def __init__(self, a, input_size, size_list):
        super(HM_LSTM, self).__init__()
        self.a = a
        self.input_size = input_size
        self.size_list = size_list

        self.cell_1 = HM_LSTMCell(self.input_size, self.size_list[0], self.size_list[1], self.a, False)
        self.cell_2 = HM_LSTMCell(self.size_list[0], self.size_list[1], None, self.a, True)

    def forward(self, inputs, hidden):
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)

        if hidden == None:
            h_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
            c_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
            z_t1 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
            h_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
            c_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
            z_t2 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
        else:
            (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = hidden
        z_one = Variable(torch.ones(1, batch_size).float().cuda(), requires_grad=False)

        h_1 = []
        h_2 = []
        z_1 = []
        z_2 = []
        for t in range(time_steps):
            h_t1, c_t1, z_t1 = self.cell_1(c=c_t1, h_bottom=inputs[:, t, :].t(), h=h_t1, h_top=h_t2, z=z_t1, z_bottom=z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c=c_t2, h_bottom=h_t1, h=h_t2, h_top=None, z=z_t2, z_bottom=z_t1)  # 0.01s used
            h_1 += [h_t1.t()]
            h_2 += [h_t2.t()]
            z_1 += [z_t1.t()]
            z_2 += [z_t2.t()]

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return torch.stack(h_1, dim=1), torch.stack(h_2, dim=1), torch.stack(z_1, dim=1), torch.stack(z_2, dim=1), hidden

class HM_Net(Module):
    def __init__(self, a, size_list, dict_size, embed_size):

        super(HM_Net, self).__init__()
        self.dict_size = dict_size
        self.size_list = size_list
        self.drop = nn.Dropout(p=0.5)
        self.norm = nn.LayerNorm([dict_size, embed_size])
        self.HM_LSTM = HM_LSTM(a, embed_size, size_list)
        self.beat = nn.ModuleList([nn.Linear(hidden_size * 2, 1), \
                                   nn.Linear(hidden_size * 2, 1)])
        self.act = nn.ModuleList([nn.Sigmoid(), nn.Sigmoid()])

    def forward(self, inputs, hidden):

        emb = self.norm(inputs)

        h_1, h_2, z_1, z_2, hidden = self.HM_LSTM(emb, hidden)  # batch_size * time_steps * hidden_size
        h_1 = self.drop(h_1)  # batch_size * time_steps * hidden_size
        h_2 = self.drop(h_2)
        h = torch.cat((h_1, h_2), 2)

        b, d = self.beat[0](h), self.beat[1](h)
        b, d = self.act[0](b), self.act[1](d)
        b, d = b.squeeze(-1), d.squeeze(-1)

        return b, d, hidden

    def init_hidden(self, batch_size):
        h_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
        c_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
        z_t1 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
        h_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
        c_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
        z_t2 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return hidden

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

    dict_size = seq_len
    embed_size = input_size
    HM_model = HM_Net(1.0, size_list, dict_size, embed_size)
    HM_model = HM_model.to(device)
    optimizer = optim.SGD(HM_model.parameters(), lr=lr, momentum=0.9)
    criteria = nn.BCELoss()

    train_arr, test_arr = [], []                     # record training loss and testing loss 

    for epoch in range(epochs):

        # training 
        HM_model.train()
        hidden = HM_model.init_hidden(batch_sz)

        bl_loss, dl_loss = 0, 0 
        for idx, (x, yb, yd) in enumerate(train_loader):

            hidden = repackage_hidden(hidden)
            x, yb, yd = x.to(device), yb.to(device), yd.to(device)
            b, d, hidden = HM_model.forward(x, hidden)

            optimizer.zero_grad()                                # update and loss 
            bl, dl = criteria(b, yb), criteria(d, yd)
            bl_loss, dl_loss = bl_loss + bl.item(), dl_loss + dl.item()
            train_loss = bl + dl * n_weights
            train_loss.backward()
            optimizer.step()

        HM_model.HM_LSTM.cell_1.a += 0.04                  # slope annealing 
        HM_model.HM_LSTM.cell_2.a += 0.04

        print('train loss = %.4f | beat loss = %.4f | downbeat loss = %.4f' % ((bl_loss + dl_loss) / idx, bl_loss / idx, dl_loss / idx))
        train_arr.append([epoch, np.round(bl_loss / idx, 4), np.round(dl_loss / idx, 4)])

        # testing 
        # bl_loss, dl_loss = 0, 0 
        # with torch.no_grad():

        #     HM_model.eval()
        #     hidden = HM_model.init_hidden(batch_sz)

        #     for idx, (x, yb, yd) in enumerate(test_loader):

        #         x, yb, yd = x.to(device), yb.to(device), yd.to(device)
        #         b, d, hidden = HM_model.forward(x, hidden)
        #         bl, dl = criteria(b, yb), criteria(d, yd)
        #         bl_loss, dl_loss = bl_loss + bl.item(), dl_loss + dl.item()

        # print('test loss = %.4f | beat loss = %.4f | downbeat loss = %.4f' % ((bl_loss + dl_loss) / idx, bl_loss / idx, dl_loss / idx))
        # test_arr.append([epoch, np.round(bl_loss / idx, 4), np.round(dl_loss / idx, 4)])

        # with open('./loss/' + folder + '_loss.npz', 'wb') as f:
        #     pickle.dump([train_arr, test_arr], f)

        model_name = './models/' + folder + '_' + str(epoch) +'.pkl'
        torch.save({
                'model_state_dict': HM_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_name)   

if __name__ == '__main__':
    main()