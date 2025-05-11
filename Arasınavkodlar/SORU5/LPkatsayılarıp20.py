from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hamming
from scipy.linalg import solve_toeplitz

# Ses dosyasını oku
sample_rate, data = wavfile.read('m18ae_40ms.wav')

# Stereo ise mono'ya dönüştür
if data.ndim > 1:
    data = data[:, 0]

# 40 ms örnek sayısı
window_duration = 0.04  # 40 ms
window_length = int(sample_rate * window_duration)

# Eğer sinyal pencere uzunluğundan uzunsa, ilk pencereyi al
if len(data) >= window_length:
    windowed_signal = data[:window_length]
else:
    raise ValueError("Sinyal 40 ms'den daha kısa!")

# Hamming penceresi uygula
hamming_window = hamming(window_length, sym=False)
windowed_signal = windowed_signal * hamming_window

# Özilinti (autocorrelation) hesapla
def autocorrelation(x, order):
    r = np.correlate(x, x, mode='full')
    mid = len(r) // 2
    return r[mid:mid + order + 1]

# LP derecesi
p = 20
r = autocorrelation(windowed_signal, p)

# Yule-Walker denklemlerini çöz (özilinti yöntemi)
R = r[:p]
rhs = r[1:p+1]
lp_coeffs = solve_toeplitz((R, R), rhs)
lp_coeffs = np.insert(-lp_coeffs, 0, 1)  # a[0] = 1

# Sonuçları yazdır
print("LP katsayıları (p=20):")
print(lp_coeffs)
