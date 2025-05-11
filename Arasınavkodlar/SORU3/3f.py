import numpy as np
import matplotlib.pyplot as plt

# Parametreler
alpha = 0.95
fs = 1000  # Örnekleme frekansı (Hz)
T = 1.0    # Toplam süre (saniye)
N = int(T * fs)  # Örnek sayısı

# 1. ˜g[n] sinyalini oluşturma (zaman ters çevrilmiş glottal flow)
n = np.arange(-N//2, N//2)  # -500 ile 499 arası
u_minus_n = (n <= 0).astype(float)  # u[-n] = 1 (n ≤ 0 için)
g_tilde = (alpha**-n * u_minus_n) * (alpha**-n * u_minus_n)  # β = α = 0.95 varsayıldı

# 2. Periyodik impuls dizilerini oluşturma (100 Hz ve 220 Hz)
def create_periodic_impulses(f0, fs, T):
    t = np.arange(0, T, 1/fs)
    period = int(fs / f0)  # Periyot (örnek sayısı)
    impulses = np.zeros_like(t)
    impulses[::period] = 1  # Her periyotta 1
    return impulses

impulse_100Hz = create_periodic_impulses(100, fs, T)
impulse_220Hz = create_periodic_impulses(220, fs, T)

# 3. Konvolüsyon işlemi
conv_result_100Hz = np.convolve(g_tilde, impulse_100Hz, mode='full')
conv_result_220Hz = np.convolve(g_tilde, impulse_220Hz, mode='full')

# 4. Grafikler
plt.figure(figsize=(15, 8))

# 100 Hz için
plt.subplot(2, 1, 1)
plt.plot(conv_result_100Hz[:1000], 'b-')
plt.title('Periyodik Glottal-Flow (100 Hz Temel Frekans)')
plt.xlabel('Örnek İndisi')
plt.ylabel('Genlik')
plt.grid(True)

# 220 Hz için
plt.subplot(2, 1, 2)
plt.plot(conv_result_220Hz[:1000], 'r-')
plt.title('Periyodik Glottal-Flow (220 Hz Temel Frekans)')
plt.xlabel('Örnek İndisi')
plt.ylabel('Genlik')
plt.grid(True)

plt.tight_layout()
plt.show()