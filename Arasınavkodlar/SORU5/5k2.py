from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hamming
from scipy.signal  import lfilter
from scipy.linalg import solve_toeplitz

# Ses dosyasını oku ve işle
sample_rate, data = wavfile.read('m18ae_40ms.wav')
if data.ndim > 1:
    data = data[:, 0]

# Parametreler
window_duration = 0.04  # 40 ms
window_length = int(sample_rate * window_duration)
T0 = 0.008  # Periyot

# 1. Orijinal Sinyal İşlemleri
# --------------------------------------------
signal_segment = data[:window_length]
hamming_window = hamming(window_length, sym=False)
windowed_signal = signal_segment * hamming_window

# Orijinal spektrum
n = window_length
fft_orig = np.fft.fft(windowed_signal, n)
freq = np.fft.fftfreq(n, d=1/sample_rate)
positive_mask = freq >= 0
freq_axis = freq[positive_mask]
log_mag_orig = 20*np.log10(np.abs(fft_orig[positive_mask]) + 1e-12)

# 2. Sentezlenmiş Sinyal İşlemleri
# --------------------------------------------
# LP parametreleri
lp_coeffs = np.array([
    1.0, -2.04321021, 1.80736675, -0.80876211, 0.13234092,
    -0.94948854, 1.82718531, -1.38800248, 0.04252149, 0.87636789,
    -0.29918086, -0.36054449, 0.42635847, 0.07957481, -0.34242757,
    0.10404746, -0.04071762, -0.02686721, -0.02489985, 0.03078988,
    0.05252582
])
A = 0.03621

# Impuls dizisi oluştur
T0_samples = int(T0 * sample_rate)
impulse_train = np.zeros(window_length)
impulse_train[::T0_samples] = 1

# Sentezle ve spektrum hesapla
synthesized = lfilter([A], lp_coeffs, impulse_train)
fft_synth = np.fft.fft(synthesized, n)
log_mag_synth = 20*np.log10(np.abs(fft_synth[positive_mask]) + 1e-12)

# 3. Birleşik Grafik
# --------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(freq_axis, log_mag_orig, 'b', linewidth=1, label='Orijinal Sinyal (Pencerelenmiş)')
plt.plot(freq_axis, log_mag_synth, 'r--', linewidth=1.5, label='Sentezlenmiş Sinyal (H(ω))')

plt.xlim(0, sample_rate/2)
plt.xlabel('Frekans (Hz)', fontsize=12)
plt.ylabel('Genlik (dB)', fontsize=12)
plt.title(f'Spektral Karşılaştırma (T₀ = {window_duration*1000} ms)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()