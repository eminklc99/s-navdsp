import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ses dosyasını okuma ve Hamming penceresi ile pencereleme
fs, data = wavfile.read('m18ae_40ms.wav')
N = len(data)
hamming_window = np.hamming(N)
windowed_data = data * hamming_window

# Özilinti fonksiyonunu τ = -639, ..., 639 için hesaplama
autocorrelation = np.correlate(windowed_data, windowed_data, mode='full')
tau = np.arange(-N + 1, N)  # τ = -639, ..., 639

plt.figure(figsize=(12, 5))
plt.plot(tau, autocorrelation)
plt.xlabel('τ (Örnek)')
plt.ylabel('Özilinti Değeri')
plt.title('Pencerelenmiş Sinyalin Özilinti Fonksiyonu')
plt.grid()
plt.xlim(-N, N)
plt.show()