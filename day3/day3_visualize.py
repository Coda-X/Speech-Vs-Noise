import numpy as np  
import librosa, librosa.display, matplotlib.pyplot as plt
path = "data/processed/noise/dogbark1_3s.wav"
y, sr = librosa.load(path, sr=None)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
S_db = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
plt.title("Mel Spectrogram - Dog Bark (3s)")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()
