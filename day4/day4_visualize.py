import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

speech_path = "data/processed/speech/123_3s.wav"
noise_path = "data/processed/noise/dogbark1_3s.wav"

def show_mfcc(path, title):
    y, sr = librosa.load(path, sr=None, mono=True)
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
    plt.figure(figsize=(9, 3.8))
    librosa.display.specshow(M, x_axis="time", sr=sr)
    plt.colorbar(format="%+0.0f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

show_mfcc(noise_path, "MFCC - Dog Bark (3s)")
show_mfcc(speech_path, "MFCC - Speech (3s)")


    
