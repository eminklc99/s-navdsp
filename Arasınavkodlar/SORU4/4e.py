import numpy as np
from scipy.io import wavfile

# Ses dosyasını okuma ve Hamming penceresi ile pencereleme
fs, data = wavfile.read('m18ae_40ms.wav')
N = len(data)
hamming_window = np.hamming(N)
windowed_data = data * hamming_window

# Enerji hesaplama (Denklem 7)
E = np.sum(windowed_data**2)
print(f"Pencerelenmiş Sinyalin Enerjisi (E): {E:.2f}")

# Özilinti fonksiyonunu hesaplama
autocorrelation = np.correlate(windowed_data, windowed_data, mode='full')

# r[0] değeri (tau=0, dizinin ortasındaki değer)
r0 = autocorrelation[N - 1]
print(f"Özilinti Fonksiyonu r[0] Değeri: {r0:.2f}")

# Pencerelenmemiş sinyalin enerjisi
E_raw = np.sum(data**2)
print(f"Pencerelenmemiş Sinyalin Enerjisi: {E_raw:.2f}")