import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

with open('./output/tmp/blstm/test.npz', 'rb') as f:            # take BLSTM for example 
    blstm = pickle.load(f)
blstm = np.concatenate(blstm[0])

with open('./input/npz/test.npz', 'rb') as f:
    data = pickle.load(f)
x = np.concatenate(data[0], axis=0)

start = 1200 * 13 - 400 
beat_tmp = blstm[start:start+1200]
lstm_arr = []
for i in range(len(beat_tmp)):
    if beat_tmp[i] >= 0.30:
        lstm_arr.append(i)

tmp = copy.copy(x[start:start+1200, 88:88*2])
where_0, where_1 = np.where(tmp == 0), np.where(tmp == 1)
tmp[where_0], tmp[where_1] = 1, 0

sz = 17
gt_style = '--'
plt.rc('font', size=sz)          
plt.rc('axes', titlesize=sz)     
plt.rc('axes', labelsize=sz)    
plt.rc('xtick', labelsize=sz)   
plt.rc('ytick', labelsize=sz)   
plt.rc('legend', fontsize=sz)   
plt.rc('figure', titlesize=sz) 

f, (ax_p) = plt.subplots(1, 1, sharex=True, figsize = (20, 4))
ax_p.imshow(tmp.transpose(), interpolation='nearest', origin='lower', aspect='auto', cmap='hot')
for i in range(len(lstm_arr)):
    ax_p.axvline(x=lstm_arr[i], alpha=0.9, linestyle=gt_style, \
                  color='r', linewidth=2.0, dashes=(5, 1)) 
ax_p.set_yticks(np.arange(0, 88, 87))
ax_p.tick_params(axis='y', which='major', pad=15)

scale = 0.01
ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale))
ax_p.xaxis.set_major_formatter(ticks)
ax_p.xaxis.set_tick_params(pad=12)
ax_p.set_xticks(np.arange(0.0, 1201, 600))
plt.xlabel('time (sec)')
plt.savefig("./figure/test.png")

