import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parametreler
r = 1
theta = np.pi / 4

# Transfer fonksiyonu katsayıları
b = [1]  # Pay
a = [1, -2 * r * np.cos(theta), r**2]  # Payda

# Frekans Cevabı (Bode için)
w, h = signal.freqz(b, a, worN=8000)
frequency_hz = w / (2 * np.pi)  # Normalize frekans → Hz (Fs=1 varsayılarak)
magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
phase_deg = np.degrees(np.angle(h))

# Impuls Cevabı
n = 30
impulse = np.zeros(n)
impulse[0] = 1
response = signal.lfilter(b, a, impulse)

# Grafikler
plt.figure(figsize=(12, 12))

# Bode Genlik Cevabı
plt.subplot(3, 1, 1)
plt.semilogx(frequency_hz, magnitude_db)
plt.title('Bode Diyagramı - Genlik Cevabı (r=1, θ=π/4)', fontsize=12)
plt.ylabel('Genlik (dB)', fontsize=10)
plt.xlim(0.01, 0.5)  # X ekseni sınırları
plt.xticks([0.01, 0.1, 0.5], ['0.01', '0.1', '0.5'], fontsize=8)  # Önemli frekanslar
plt.grid(True, which="both", ls="--", alpha=0.6)

# Bode Faz Cevabı
plt.subplot(3, 1, 2)
plt.semilogx(frequency_hz, phase_deg)
plt.title('Bode Diyagramı - Faz Cevabı', fontsize=12)
plt.ylabel('Faz (Derece)', fontsize=10)
plt.xlabel('Frekans (Hz)', fontsize=10)
plt.xlim(0.01, 0.5)
plt.xticks([0.01, 0.1, 0.5], ['0.01', '0.1', '0.5'], fontsize=8)
plt.grid(True, which="both", ls="--", alpha=0.6)

# Impuls Cevabı
plt.subplot(3, 1, 3)
plt.stem(np.arange(n), response, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title('Impuls Cevabı', fontsize=12)
plt.xlabel('Örnekler (n)', fontsize=10)
plt.ylabel('Genlik', fontsize=10)
plt.grid(True, alpha=0.6)

plt.tight_layout()
plt.show()