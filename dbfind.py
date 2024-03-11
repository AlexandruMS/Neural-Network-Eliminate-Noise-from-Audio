import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
#file_path = 'data_clean/0-clean.wav'
# file_path = 'data_noise/0-noise.wav'
# file_path = 'data_conn/mixed_1.wav'
file_path = 'data_con-30/mixed_1.wav'
y, sr = librosa.load(file_path)

# Plot the waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.show()

# Plot the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()
