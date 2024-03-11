from fileinput import filename
from flask import Flask, render_template, request, redirect, url_for
import os
from flask import session
from SourceSeparation import MyModel, compute_rms, stft_transform, istft_transform
import torch
import soundfile as sf
import numpy as np
import resampy as rs




data, samplerate = sf.read('filename')