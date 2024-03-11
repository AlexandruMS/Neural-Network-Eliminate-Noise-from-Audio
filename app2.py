from fileinput import filename
from flask import Flask, render_template, request, redirect, url_for
import os
from flask import session
from SourceSeparation import MyModel, compute_rms, stft_transform, istft_transform
import torch
import soundfile as sf
import numpy as np
import resampy as rs
import time
device = torch.device("cuda")


def compute_gain(data, dB):
    data_rms = compute_rms(data)
    Gain = dB - 10 * np.log10(data_rms)
    gain = 10 ** (Gain / 10)

    return gain


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESS_FOLDER = 'static/proces'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESS_FOLDER'] = PROCESS_FOLDER
app.config['SESSION_TYPE'] = 'f'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index2.html', filename=None, file_format=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    if os.path.exists('static/process/result.wav'):
        os.remove('static/process/result.wav')

    time.sleep(1)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(filename)

        session['f'] = filename

        file_format = process_file(filename)

        return render_template('index2.html', filename=file.filename, file_format=file_format)

    return 'Invalid file format.'


def process_file(filename):
    _, file_extension = os.path.splitext(filename)
    return file_extension.strip('.')


@app.route('/proces', methods=['POST'])
def proces():
    path = session.get('f', None)

    if os.path.exists('static/process/result.wav'):
        os.remove('static/process/result.wav')

    if not path:
        return 'Filename not found in session.'

    path_sound = path
    data, samplerate = sf.read(path_sound)

    data = rs.resample(data, samplerate, 48000)
    length = len(data)
    gain = compute_gain(data, -20)
    data = gain * data

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

    sf.write('static/process/result.wav', data, 48000)
    file_format = process_file(path)
    file_name = os.path.basename(path)
    return render_template('index2.html', filename=file_name, file_format=file_format)


if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True)
