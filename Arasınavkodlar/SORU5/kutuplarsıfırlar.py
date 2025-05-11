import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal.windows import hamming
from scipy.signal import lfilter
# Ses dosyasını oku
sr, data = wavfile.read("m18ae_40ms.wav")
if data.ndim > 1:
    data = data[:, 0]

# 40 ms'lik pencere
duration = 0.04
N = int(sr * duration)
signal = data[:N]

# Hamming pencere uygula
windowed_signal = signal * hamming(N, sym=False)

# LP katsayıları np.array şeklinde
lp_coeffs = np.array([
    1.0, -2.04321021,  1.80736675, -0.80876211,  0.13234092,
   -0.94948854,  1.82718531, -1.38800248,  0.04252149,  0.87636789,
   -0.29918086, -0.36054449,  0.42635847,  0.07957481, -0.34242757,
    0.10404746, -0.04071762, -0.02686721, -0.02489985,  0.03078988,
    0.05252582
])
p = len(lp_coeffs) - 1

# Tahmin edilen sinyal ŝ[n]
s_hat = np.zeros_like(windowed_signal)
for n in range(p, len(windowed_signal)):
    s_hat[n] = -np.sum(lp_coeffs[1:] * windowed_signal[n-p:n][::-1])

# Hata ve kazanç A
error_signal = windowed_signal - s_hat
A = np.sqrt(np.sum(error_signal[p:]**2) / (N - p))
print(f"Kazanç katsayısı A: {A:.5f}")

# Impuls cevabı
impulse_len = 100
impulse = np.zeros(impulse_len)
impulse[0] = 1
h = lfilter([A], lp_coeffs, impulse)

# Zaman vektörü
t = np.arange(impulse_len) / sr
"""
# Grafik
plt.figure(figsize=(10, 4))
plt.stem(t, h, basefmt=" ")
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.title("H(z) Süzgecinin İmpuls Cevabı")
plt.grid(True)
plt.tight_layout()
plt.show()
"""

from scipy.signal import tf2zpk
import matplotlib.pyplot as plt

# Numerator ve denominator tanımı
b = [A]                 # Numerator (sıfırlar) -> sadece kazanç
a = lp_coeffs           # Denominator (kutuplar) -> LP polinomu

# Kutup ve sıfırları bul
z, p, k = tf2zpk(b, a)

# Birim çember çizimi
theta = np.linspace(0, 2*np.pi, 512)
unit_circle = np.exp(1j * theta)

# Grafik
plt.figure(figsize=(6, 6))
plt.plot(np.real(unit_circle), np.imag(unit_circle), 'k--', label='Birim Çember')
plt.plot(np.real(z), np.imag(z), 'go', label='Sıfırlar')   # Sıfırlar
plt.plot(np.real(p), np.imag(p), 'rx', label='Kutuplar')   # Kutuplar
plt.title('Kutup-Sıfır Diyagramı (Pole-Zero Plot)')
plt.xlabel('Gerçek')
plt.ylabel('Sanal')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

is_stable = np.all(np.abs(p) < 1)
print("Sistem kararlı mı?", "Evet" if is_stable else "Hayır")