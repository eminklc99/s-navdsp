from scipy.io import wavfile
import numpy as np

# Ses dosyasını oku
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

p = 20  # LP derecesi
n = len(windowed_signal)

# Özilinti (autocorrelation) hesapla
r = np.zeros(p + 1)
for k in range(p + 1):
    r[k] = np.sum(windowed_signal[k:n] * windowed_signal[0:n - k])
    
    # Levinson-Durbin algoritması
a = np.zeros(p + 1)
e = np.zeros(p + 1)

a[0] = 1
e[0] = r[0]

for k in range(1, p + 1):
    lambda_val = -np.sum(a[:k] * r[k:0:-1]) / e[k-1]
    a[1:k+1] = a[1:k+1] + lambda_val * a[k-1::-1]
    a[k] = lambda_val
    e[k] = e[k-1] * (1 - lambda_val**2)

# LP katsayıları (α1, α2, ..., α20)
alpha = -a[1:]
print("LP Katsayıları (αk):", alpha)