import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

sample_rate, data = wavfile.read('m18ae_40ms.wav')

# Mono'ya dönüştür (gerekirse)
if len(data.shape) > 1:
    data = data[:, 0]

# 40 ms'lik pencere boyutu (örnek sayısı)
window_length = int(0.04 * sample_rate)  # 0.04 saniye * örnekleme hızı
start = 0  # Sinyal zaten 40 ms ise tamamını al
end = start + window_length
signal_segment = data[start:end]

# Hamming penceresi uygula
hamming_window = np.hamming(len(signal_segment))
windowed_signal = signal_segment * hamming_window
p = 20
alpha_custom = np.array([
    2.04329167, -1.80745716, 0.80866572, -0.13175562, 0.9482427,
    -1.82588242, 1.38733435, -0.04238493, -0.87668338, 0.29998738,
    0.35980254, -0.42623509, -0.07922955, 0.34230059, -0.10438987,
    0.04121287, 0.02656084, 0.0249514, -0.03081125, -0.05249107
])


# Tahmin edilen sinyali hesapla
s_hat = np.zeros_like(windowed_signal)
for n in range(p, len(windowed_signal)):
    s_hat[n] = np.sum(alpha_custom * windowed_signal[n - p : n][::-1])

# Zaman vektörü
t = np.linspace(0, 0.04, len(windowed_signal))  # 40 ms = 0.04 saniye

plt.figure(figsize=(12, 6))
plt.plot(t, windowed_signal, label='Orijinal Sinyal', color='blue', alpha=0.7)
plt.plot(t, s_hat, label='Tahmin Edilen Sinyal (ŝ[n])', color='red', linestyle='--')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.title('Orijinal vs. Tahmin Edilen Sinyal (p=20)')
plt.legend()
plt.grid(True)
plt.show()