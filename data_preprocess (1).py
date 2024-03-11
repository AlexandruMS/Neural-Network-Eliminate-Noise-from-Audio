import soundfile as sf
import resampy as rs
import numpy as np
import resampy
import os
import glob as gb

if not os.path.exists('./data_clean'):
    print('Create directory....')
    os.mkdir('data_clean')

data = []
index = 0
for files in gb.glob('./clean/LibriSpeech/dev-clean/*/*/*.flac', recursive=True):
    d, samp = sf.read(files)
    d = resampy.resample(d, samp, 48000)
    if len(data) < 480000:
        data.extend(d)
    else:
        print(index)
        sf.write('./data_clean/'+str(index)+'-clean.wav', data[0:480000], 48000)
        index += 1
        data = []
        data.extend(d)
    print(files)
