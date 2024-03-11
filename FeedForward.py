from SourceSeparation import MyModel, compute_rms, stft_transform, istft_transform
import torch
import soundfile as sf
import numpy as np
import resampy as rs

device = torch.device("cuda")

def compute_gain(data, dB):
    data_rms = compute_rms(data)
    Gain = dB - 10 * np.log10(data_rms)
    gain = 10 ** (Gain / 10)

    return gain


if __name__ == "__main__":

    path_sound = "ENG_M.wav"
    data, samplerate = sf.read(path_sound)

    data = rs.resample(data, samplerate, 48000)
    length = len(data)
    gain = compute_gain(data, -20)
    data = gain * data
    '''
    
    
    data_clean, _ = sf.read("data_clean/1-clean.wav")
    data_noise, _ = sf.read("0-noise.wav")

    gain_clean = compute_gain(data_clean, -20)
    gain_noise = compute_gain(data_clean, -20)

    data = gain_clean * data_clean + gain_noise * data_noise
    
    sf.write("input.wav", data/gain_clean, 48000)

    print(data_clean, len(data))
    '''
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    magnitude, phase = stft_transform(data_tensor)
    magnitude = torch.unsqueeze(magnitude, 0)
    phase = torch.unsqueeze(phase, 0)

    model = MyModel().to(device)
    model.load_state_dict(torch.load("model_GPT_1000_100_20.pth"))

    magnitude = model(magnitude)
    magnitude = torch.squeeze(magnitude, 0)
    phase = torch.squeeze(phase, 0)

    data_tensor = istft_transform(magnitude, phase, length)
    data = data_tensor.to('cpu').detach().numpy()

    data /= gain


    print(data, len(data))


    sf.write("result.wav", data, 48000)