import librosa, librosa.display
import matplotlib.pyplot as plt

file_path = "data/noise/dogbark1.wav"

audio, sr = librosa.load(file_path, sr=None)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title("Waveform of Dog Bark")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

