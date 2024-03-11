import soundfile as sf
import resampy as rs
import glob as gb
import numpy as np
import resampy
import os

if not os.path.exists('./data_noise'):
    print('Create directory....')
    os.mkdir('data_noise')

data = []
index = 0
for file in gb.glob('./noise/TUT-acoustic-scenes-2017-development/audio/*.wav', recursive=True):
    d, samp = sf.read(file)
    d = (d[:, 0] + d[:, 1])/2
    d = resampy.resample(d, samp, 48000)
    if len(data) < 480000:
        data.extend(d)
    else:
        print(index)
        sf.write('./data_noise/'+str(index)+'-noise.wav', data[0:480000], 48000)
        index += 1
        data = d
    print(file)
