import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal.windows import hamming
from scipy.linalg import solve_toeplitz

# WAV dosyasını oku
sample_rate, data = wavfile.read('m18ae_40ms.wav')

# Mono'ya çevir
if data.ndim > 1:
    data = data[:, 0]

# 40 ms'lik pencere
window_duration = 0.04
window_length = int(sample_rate * window_duration)
signal = data[:window_length]

# Hamming pencere uygula
windowed_signal = signal * hamming(window_length, sym=False)

# Özilinti yöntemiyle LP katsayılarını hesapla
def autocorrelation(x, order):
    r = np.correlate(x, x, mode='full')
    mid = len(r) // 2
    return r[mid:mid + order + 1]

p = 20
r = autocorrelation(windowed_signal, p)
R = r[:p]
rhs = r[1:p+1]
lp_coeffs = solve_toeplitz((R, R), rhs)
lp_coeffs = np.insert(-lp_coeffs, 0, 1)  # a[0] = 1

# Tahmin edilen sinyali hesapla (denklem 9)
s_hat = np.zeros_like(windowed_signal)
for n in range(p, len(windowed_signal)):
    s_hat[n] = -np.sum(lp_coeffs[1:] * windowed_signal[n - p:n][::-1])
# Grafik çizimi
t = np.linspace(0, window_duration, window_length)

plt.figure(figsize=(12, 4))
plt.plot(t, windowed_signal, label='Orijinal Sinyal', color='blue')
plt.plot(t, s_hat, label='Tahmin Edilen Sinyal', color='red', linestyle='--')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.title('Orijinal ve Tahmin Edilen Sinyal (LP)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
