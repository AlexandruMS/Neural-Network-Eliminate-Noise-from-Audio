import torch
from torch import nn
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

device = torch.device("cuda")
n_fft = 1024
hop_length = 512


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=513, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.bnorm = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128, 513)
        self.sigmoid = nn.Sigmoid()

        '''
        self.lstm = nn.LSTM(513, 256, num_layers=2, batch_first=True,bidirectional=True)
        self.bnorm = nn.BatchNorm1d(256)
        self.lin1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(128, 513)
        self.sigmoid = nn.Sigmoid()
        '''

    def forward(self, x):
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.bnorm(x)
        x = x.transpose(1, 2)
        # x = x.transpose(1, 2) # Take the output from the last time step
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        x = self.sigmoid(x)
        x = x.transpose(1, 2)
        '''
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.bnorm(x)
        x = x.transpose(1, 2)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        x = x.transpose(1, 2)
        '''
        return x


def compute_rms(data):
    return np.sqrt(np.mean(data**2))


def compute_gain(clean, noise):
    clean_rms = compute_rms(clean)
    noise_rms = compute_rms(noise)

    G_clean = -20 - 10 * np.log10(clean_rms)
    # G_noise = random.randint(-20, -11) - 10 * np.log10(noise_rms)
    G_noise = -20 - 10 * np.log10(noise_rms)

    g_clean = 10**(G_clean / 10)
    g_noise = 10**(G_noise / 10)

    return g_clean, g_noise

def toCSV():
    # get a list of all files in the directory and save to a csv
    with open('clean.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name_file'])
        for data in glob.glob('./data_clean/*.wav'):
            name = data.removeprefix('./data_clean\\')
            writer.writerow([name])
    with open('noise.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name_file'])
        for data in glob.glob('./data_noise/*.wav'):
            name = data.removeprefix('./data_noise\\')
            writer.writerow([name])


# class to create a custom dataset loader

class CustomSoundDataset(Dataset):
    def __init__(
        self, voice_annotation_file, noise_annotation_file, voice_dir, noise_dir
    ):
        self.voice_name = pd.read_csv(voice_annotation_file)
        self.noise_name = pd.read_csv(noise_annotation_file)

        self.voice_dir = voice_dir
        self.noise_dir = noise_dir


    # lungimea datasetului = nr. de samples = 1000

    def __len__(self):
        number_of_example = 1000
        return number_of_example

    def __getitem__(self, index):
        path_clean = os.path.join(self.voice_dir, self.voice_name['name_file'].iloc[index])
        path_noise = os.path.join(self.noise_dir, self.noise_name['name_file'].iloc[random.randint(0, self.noise_name['name_file'].size-1)])
        clean, _ = sf.read(path_clean)
        noise, _ = sf.read(path_noise)

        g_clean, g_noise = compute_gain(clean, noise)
        clean = g_clean * clean
        noise = g_noise * noise

        x = clean + noise
        y = clean

        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)

        X_mag, X_phase = stft_transform(X)
        Y_mag, Y_phase = stft_transform(Y)

        X_mag = X_mag.to(device)
        Y_mag = Y_mag.to(device)

        return X_mag, Y_mag, X_phase, Y_phase


# stft transform - use it if needed
def stft_transform(signal):

    spectrogram = torch.stft(signal, n_fft, hop_length=hop_length, center=True, return_complex=True)
    magnitude = spectrogram.abs()
    phase = spectrogram.angle()

    return magnitude, phase


# istft transform - use it if needed
def istft_transform(magnitude, phase, length):

    spectrogram = torch.polar(magnitude, phase)
    signal = torch.istft(spectrogram, n_fft, hop_length=hop_length, center=True, length=length)
    return signal


# use if needed
def compute_magnitude(complex_signal):
    pass


def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = []
    for batch, (X_mag, Y_mag, _, _) in enumerate(dataloader):
        # data preprocessing and model forward
        optimizer.zero_grad()
        Pred_mag = model(X_mag)
        loss = loss_fn(Pred_mag, Y_mag)
        total_loss.append(loss)
        # pred, targets = None, None
        # loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
    # to be completed
    print(f"Loss this epoch is: {sum(total_loss)/len(total_loss)}")


if __name__ == "__main__":

    csd = CustomSoundDataset('clean.csv','noise.csv', './data_clean', './data_noise')

    train_dataloader = DataLoader(dataset=csd, batch_size=64, shuffle=False)
    model = MyModel().to(device)

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000001)
    epochs = 100

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        train(model, train_dataloader, loss_fn, optimizer)
        torch.save(model.state_dict(), "model_GPT_1000_100_20.pth")
    print("Train done!")

    '''
    loss_fn = nn.MSELoss()

    for batch, (X_mag, Y_mag, _, _) in enumerate(train_dataloader):
        Pred_mag = model(X_mag)
        print(Pred_mag.size())

        # total_loss += loss

        break
    
   
    total_loss = torch.tensor(total_loss, dtype=torch.float32)
    total_loss /= 1000

    print(total_loss)

    csd = CustomSoundDataset()
    train_dataloader = DataLoader(csd, batch_size=10, shuffle=False)
    model = MyModel.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)
    epochs = 20

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(model, train_dataloader, loss_fn, optimizer)
        torch.save(model.state_dict(), "model.pth")
    print("Train done!")
    '''
    '''
    #recostructie test
    a = torch.randn(480000)
    print(a)
    print(a.size())
    abs, angle = stft_transform(a)

    
    b = istft_transform(abs, angle)
    print(b)
    print(b.size())
    '''
