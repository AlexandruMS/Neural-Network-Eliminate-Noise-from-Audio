import soundfile as sf
import resampy as rs
import numpy as np
import glob as gb

for file in gb.glob('./data_noise/*.wav', recursive=True):
    data, samp = sf.read(file)
    print(file, len(data))
