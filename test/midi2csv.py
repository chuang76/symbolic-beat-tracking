import glob
import numpy as np
import pandas as pd
import pretty_midi

file_list = np.sort(glob.glob('./input/*.mid', recursive=False)).tolist()

for file_idx in range(len(file_list)):
    data = pretty_midi.PrettyMIDI(file_list[file_idx], resolution=441)
    name = file_list[file_idx].split('/')[2].split('.')[0]
    midi_list = []
    for instrument in data.instruments:
        for note in instrument.notes:
            start = int(note.start * 44100)
            end = int(note.end * 44100)
            pitch = note.pitch
            midi_list.append([start, end, pitch])

    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))
    df = pd.DataFrame(midi_list, columns=['start_time', 'end_time', 'note'])
    df.to_csv('./input/' + name +'.csv')
