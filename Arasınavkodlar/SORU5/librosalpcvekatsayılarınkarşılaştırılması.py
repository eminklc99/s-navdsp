import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hamming
from scipy.linalg import solve_toeplitz
import librosa

# Ses dosyasını oku
sample_rate, data = wavfile.read('m18ae_40ms.wav')

# Stereo ise mono'ya dönüştür
if data.ndim > 1:
    data = data[:, 0]

# 40 ms uzunlukta pencere seç
window_duration = 0.04  # saniye
window_length = int(sample_rate * window_duration)
signal = data[:window_length]

# Hamming penceresi uygula
windowed_signal = signal * hamming(window_length, sym=False)

# Özilinti yöntemiyle LP katsayıları (elle)
def autocorrelation(x, order):
    r = np.correlate(x, x, mode='full')
    mid = len(r) // 2
    return r[mid:mid + order + 1]

p = 20
r = autocorrelation(windowed_signal, p)
R = r[:p]
rhs = r[1:p+1]
lp_custom = solve_toeplitz((R, R), rhs)
lp_custom = np.insert(-lp_custom, 0, 1)

# Librosa ile LPC
# Not: librosa sinyali float32 istiyor
windowed_signal_float = windowed_signal.astype(np.float32)
lp_librosa = librosa.lpc(windowed_signal_float, order=p)

# Karşılaştırma
print("Elle hesaplanan LP katsayıları:")
print(lp_custom)
print("\nlibrosa.lpc() ile hesaplanan LP katsayıları:")
print(lp_librosa)

# Farkların büyüklüğü
diff = np.abs(lp_custom - lp_librosa)
print("\nKatsayılar arasındaki mutlak fark:")
print(diff)
