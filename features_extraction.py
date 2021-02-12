import numpy as np
import pandas as pd
import librosa
import librosa.display

sr = 22050
path = "genres_original/"

# file = "genres_original/blues/blues.00000.wav"
# with wave.open(file, 'rb') as f:
#     framerate = f.getframerate()
#
# Signal, sr = librosa.load(file, sr=22050)
#
# plt.figure(figsize=(15,5))
# librosa.display.waveplot(Signal, sr=sr)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title("Classical music signal")
# plt.show()

df = pd.DataFrame()

files = librosa.util.find_files(path, ext=['wav'])
files = np.array(files)
files = np.delete(files, 554)

df = pd.DataFrame(data={'filename': [], 'genre': [],
                        'zcr_mean': [], 'zcr_var': [],
                        'spectral_centroid_mean': [], 'spectral_centroid_var': [],
                        'spectral_rolloff_mean': [], 'spectral_rolloff_var': [],
                        'chroma_mean': [], 'chroma_var': [],
                        'mfcc1_mean': [], 'mfcc1_var': [], 'mfcc2_mean': [], 'mfcc2_var': [],
                        'mfcc3_mean': [], 'mfcc3_var': [], 'mfcc4_mean': [], 'mfcc4_var': [],
                        'mfcc5_mean': [], 'mfcc5_var': [], 'mfcc6_mean': [], 'mfcc6_var': [],
                        'mfcc7_mean': [], 'mfcc7_var': [], 'mfcc8_mean': [], 'mfcc8_var': [],
                        'mfcc9_mean': [], 'mfcc9_var': [], 'mfcc10_mean': [], 'mfcc10_var': [],
                        'mfcc11_mean': [], 'mfcc11_var': [], 'mfcc12_mean': [], 'mfcc12_var': [],
                        'mfcc13_mean': [], 'mfcc13_var': []})

signals = []
for i, f in enumerate(files):
    x, sr = librosa.load(files[i], sr=22050)
    signals.append(x)

for i, f in enumerate(files):
    t = f.split('\\')
    x = signals[i]
    zcr = librosa.feature.zero_crossing_rate(x)
    s_centroid = librosa.feature.spectral_centroid(x, sr=sr)
    s_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
    chroma = librosa.feature.chroma_stft(x, sr=sr)
    MFCCS = librosa.feature.mfcc(x, sr=sr)
    features = [t[-1], t[-2],
                np.mean(zcr), np.var(zcr),
                np.mean(s_centroid), np.var(s_centroid),
                np.mean(s_rolloff), np.var(s_rolloff),
                np.mean(chroma), np.var(chroma)]

    #Extract mean and variance of the first 13 MFCCs
    for c in range(0, 13):
        features = features + [np.mean(MFCCS[c]), np.var(MFCCS[c])]

    df.loc[i] = features

# x, sr = librosa.load(files[0], sr=22050)
#
# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# plt.title("Spectogram")
# plt.colorbar()
