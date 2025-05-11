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

# Sinyal kontrolü ve penceresini al
if len(data) >= window_length:
    signal_segment = data[:window_length]
else:
    raise ValueError("Sinyal 40 ms'den daha kısa!")

# Hamming penceresi uygula
hamming_window = hamming(window_length, sym=False)
windowed_signal = signal_segment * hamming_window

# Logaritmik Genlik Spektrumu Hesapla
n = len(windowed_signal)
fft_result = np.fft.fft(windowed_signal, n)
magnitude = np.abs(fft_result)
log_magnitude = 20 * np.log10(magnitude + 1e-12)  # dB cinsinden

# Frekans ekseni
freq = np.fft.fftfreq(n, d=1/sample_rate)
positive_mask = freq >= 0
freq_positive = freq[positive_mask]
log_magnitude_positive = log_magnitude[positive_mask]

# Grafik
plt.figure(figsize=(10, 5))
plt.plot(freq_positive, log_magnitude_positive)
plt.xlim(0, sample_rate/2)
plt.xlabel('Frekans (Hz)', fontsize=12)
plt.ylabel('Genlik (dB)', fontsize=12)
plt.title(f'Pencerelenmiş Sinyalin Logaritmik Genlik Spektrumu (T₀ = {window_duration*1000} ms)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# (Önceki kodun LP katsayı hesaplama kısmı burada devam eder...)