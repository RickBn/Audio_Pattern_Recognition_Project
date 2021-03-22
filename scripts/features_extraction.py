import numpy as np
import pandas as pd
from scripts.functions import energy_entropy
import librosa
import librosa.display

sr = 22050
path = "data/genres_original/"

df = pd.DataFrame()

files = librosa.util.find_files(path, ext=['wav'])
files = np.array(files)
files = np.delete(files, 554)

df = pd.DataFrame(data={'genre': [], 'rms_mean': [], 'rms_std': [],
                        'energy_entropy_mean': [], 'energy_entropy_std': [],
                        'sp_flux_mean': [], 'sp_flux_std': [], 'mel_spectrogram_mean': [], 'mel_spectrogram_std': [],
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
    y, sr = librosa.load(files[i], sr=22050)
    signals.append(y)

for i, f in enumerate(files):
    t = f.split('\\')
    y = signals[i]
    rms = librosa.feature.rms(y)[0]
    entropy = energy_entropy(y)
    flux = librosa.onset.onset_strength(y, sr=sr)
    mel_sp = librosa.feature.melspectrogram(y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    s_centroid = librosa.feature.spectral_centroid(y, sr=sr)
    s_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
    chroma = librosa.feature.chroma_stft(y, sr=sr)
    MFCCS = librosa.feature.mfcc(y, sr=sr)

    features = [t[-2],
                np.mean(rms), np.std(rms),
                np.mean(entropy), np.std(entropy),
                np.mean(flux), np.std(flux),
                np.mean(mel_sp), np.std(mel_sp),
                np.mean(zcr), np.std(zcr),
                np.mean(s_centroid), np.std(s_centroid),
                np.mean(s_rolloff), np.std(s_rolloff),
                np.mean(chroma), np.std(chroma)]

    #Extract mean and std of the first 13 MFCCs
    for c in range(0, 13):
        features = features + [np.mean(MFCCS[c]), np.std(MFCCS[c])]

    df.loc[i] = features

df.to_csv("data/features.csv")