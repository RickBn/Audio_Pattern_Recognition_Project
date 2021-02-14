import numpy as np
import pandas as pd
import librosa
import librosa.display

sr = 22050
path = "data/genres_original/"

# file = "data/genres_original/blues/blues.00000.wav"
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

df = pd.DataFrame(data={'genre': [], 'rms_mean': [], 'rms_std': [], 'tempo': [],
                        'zcr_mean': [], 'zcr_std': [], 'spectral_centroid_mean': [], 'spectral_centroid_std': [],
                        'spectral_rolloff_mean': [], 'spectral_rolloff_std': [], 'chroma_mean': [], 'chroma_std': [],
                        'mfcc1_mean': [], 'mfcc1_std': [], 'mfcc2_mean': [], 'mfcc2_std': [],
                        'mfcc3_mean': [], 'mfcc3_std': [], 'mfcc4_mean': [], 'mfcc4_std': [],
                        'mfcc5_mean': [], 'mfcc5_std': [], 'mfcc6_mean': [], 'mfcc6_std': [],
                        'mfcc7_mean': [], 'mfcc7_std': [], 'mfcc8_mean': [], 'mfcc8_std': [],
                        'mfcc9_mean': [], 'mfcc9_std': [], 'mfcc10_mean': [], 'mfcc10_std': [],
                        'mfcc11_mean': [], 'mfcc11_std': [], 'mfcc12_mean': [], 'mfcc12_std': [],
                        'mfcc13_mean': [], 'mfcc13_std': []})

signals = []
for i, f in enumerate(files):
    x, sr = librosa.load(files[i], sr=22050)
    signals.append(x)

for i, f in enumerate(files):
    t = f.split('\\')
    x = signals[i]
    rms = librosa.feature.rms(x)
    oenv = librosa.onset.onset_strength(x, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(x)
    s_centroid = librosa.feature.spectral_centroid(x, sr=sr)
    s_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
    chroma = librosa.feature.chroma_stft(x, sr=sr)
    MFCCS = librosa.feature.mfcc(x, sr=sr)

    features = [t[-2],
                np.mean(rms), np.std(rms), tempo,
                np.mean(zcr), np.std(zcr),
                np.mean(s_centroid), np.std(s_centroid),
                np.mean(s_rolloff), np.std(s_rolloff),
                np.mean(chroma), np.std(chroma)]

    #Extract mean and std of the first 13 MFCCs
    for c in range(0, 13):
        features = features + [np.mean(MFCCS[c]), np.std(MFCCS[c])]

    df.loc[i] = features

df.to_csv("data/new.csv")

# x, sr = librosa.load(files[0], sr=22050)
#
# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# plt.title("Spectogram")
# plt.colorbar()
