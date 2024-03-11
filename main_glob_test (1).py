import os
import glob as gb
import soundfile as sf
import resampy as rs

# os.mkdir('clean')

# for path in gb.glob('./speech/dev-clean/Librispeech/dev-clean/*'):
#   path = './clean' + \
#       path.removeprefix('./speech/dev-clean/Librispeech/dev-clean')
#   print(path)
#   os.mkdir(path)

# print('Name of files')
# for path in gb.glob('./speech/dev-clean/Librispeech/dev-clean/*/*'):
#    path = './clean' + \
#        path.removeprefix('./speech/dev-clean/Librispeech/dev-clean')
#    print(path)
#    os.mkdir(path)

for name in gb.glob('./speech/dev-clean/Librispeech/dev-clean/*/*/*.flac', recursive=True):
    data_voice, samplerate_voice = sf.read(name)
    data_voice = rs.resample(data_voice, samplerate_voice, 48000)
    name = './clean' + \
        name.removeprefix("./speech/dev-clean/Librispeech/dev-clean")
    sf.write(name, data_voice, 48000)
    print(name)
