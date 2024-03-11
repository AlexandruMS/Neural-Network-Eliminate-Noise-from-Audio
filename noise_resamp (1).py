import os
import glob as gb
import soundfile as sf
import resampy as rs

# os.mkdir('noise')

for name in gb.glob('./audio/*.wav', recursive=True):
    data_noise, samplerate_noise = sf.read(name)
    data_noise = (data_noise[:, 1] + data_noise[:, 0])/2
    data_noise = rs.resample(data_noise, samplerate_noise, 48000)
    name = './noise' + \
        name.removeprefix("./audio")
    sf.write(name, data_noise, 48000)
    print(name)
