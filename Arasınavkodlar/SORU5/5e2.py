import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks, correlate
from scipy.signal.windows import hamming
from scipy.linalg import solve_toeplitz

# Ses dosyasını oku
sample_rate, data = wavfile.read('m18ae_40ms.wav')

# Mono'ya çevir
if data.ndim > 1:
    data = data[:, 0]

# 40 ms pencere
window_duration = 0.04
window_length = int(sample_rate * window_duration)
signal = data[:window_length]

# Hamming pencere
windowed_signal = signal * hamming(window_length, sym=False)

# LP katsayıları (özilinti yöntemiyle)
def autocorrelation(x, order):
    r = np.correlate(x, x, mode='full')
    mid = len(r) // 2
    return r[mid:mid + order + 1]

p = 20
r = autocorrelation(windowed_signal, p)
R = r[:p]
rhs = r[1:p+1]
lp_coeffs = solve_toeplitz((R, R), rhs)
lp_coeffs = np.insert(-lp_coeffs, 0, 1)

# LP tahmin sinyali
s_hat = np.zeros_like(windowed_signal)
for n in range(p, len(windowed_signal)):
    s_hat[n] = -np.sum(lp_coeffs[1:] * windowed_signal[n - p:n][::-1])

# Hata sinyali
error_signal = windowed_signal - s_hat

# Zaman vektörü
t = np.linspace(0, window_duration, window_length)

# Hata sinyali grafiği
plt.figure(figsize=(12, 4))
plt.plot(t, error_signal, label='Hata Sinyali e[n]')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.title('Hata Sinyali (e[n] = s[n] - ŝ[n])')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Hata sinyalinin otokorelasyonu
error_autocorr = correlate(error_signal, error_signal, mode='full')
mid = len(error_autocorr) // 2
peaks, _ = find_peaks(error_autocorr[mid:])

# En büyük ikinci tepe noktasının konumu = periyot
if len(peaks) > 0:
    lag = peaks[0]  # ilk tepe (0'dan sonra gelen en erken tepe)
    estimated_period_seconds = lag / sample_rate
    print(f"Hata sinyalinin yaklaşık periyodu: {estimated_period_seconds:.5f} saniye")
else:
    print("Tepe bulunamadı, periyot tahmini yapılamadı.")

# A kazancı hesaplama
N = len(windowed_signal)
error_energy = np.sum((error_signal[p:])**2)
A = np.sqrt(error_energy / (N - p))

print(f"LP modelinin kazanç katsayısı A: {A:.5f}")
