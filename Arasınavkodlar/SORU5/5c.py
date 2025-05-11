import librosa
import numpy as np

# Ses dosyasını oku (Librosa ile)
signal, sr = librosa.load('m18ae_40ms.wav', sr=None, mono=True)

# 40 ms'lik Hamming penceresi uygula
window_length = int(0.04 * sr)
start = 0
end = start + window_length
signal_segment = signal[start:end]
hamming_window = np.hamming(len(signal_segment))
windowed_signal = signal_segment * hamming_window

# Librosa ile LP katsayılarını hesapla (p=20)
alpha_librosa = librosa.lpc(windowed_signal, order=20)
print("Librosa LP Katsayıları:", alpha_librosa[1:])  # α0=1 olduğu için atlanır

alpha_custom = np.array([
    2.04329167, -1.80745716, 0.80866572, -0.13175562, 0.9482427,
    -1.82588242, 1.38733435, -0.04238493, -0.87668338, 0.29998738,
    0.35980254, -0.42623509, -0.07922955, 0.34230059, -0.10438987,
    0.04121287, 0.02656084, 0.0249514, -0.03081125, -0.05249107
])

# Katsayıları normalize et (Librosa işaretleri farklı olabilir)
alpha_custom = np.array(alpha_custom)
alpha_librosa = alpha_librosa[1:]  # α0=1'i çıkar

# Mutlak farkları hesapla
diff = np.max(np.abs(alpha_custom - alpha_librosa))
print("Maksimum Mutlak Fark:", diff)

if np.allclose(alpha_custom, alpha_librosa, atol=1e-3):
    print("Sonuçlar benzer (≤ 0.001 tolerans).")
else:
    print("Önemli farklar var!")