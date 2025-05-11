import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

# Ses dosyasını okuma
fs, data = wavfile.read('m18ae_40ms.wav')
N = len(data)  # 640 örnek (40 ms * 16 kHz)

# Hamming penceresini oluşturma ve sinyali pencereleme
hamming_window = np.hamming(N)
windowed_data = data * hamming_window

zaman = np.arange(N) / fs  # Zaman vektörü (saniye)

plt.figure(figsize=(12, 5))
plt.plot(zaman, windowed_data, color='orange')
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.title('Hamming Pencerelenmiş Sinyal')
plt.grid()
plt.show()

# Tepe noktalarını bulma (threshold ve distance parametreleri ayarlanabilir)
peaks, _ = find_peaks(windowed_data, height=0.5*max(windowed_data), distance=50)

# Tepe noktaları arasındaki ortalama zaman farkı
peak_times = zaman[peaks]
time_diff = np.diff(peak_times)
avg_period = np.mean(time_diff)

print(f"Ortalama Periyot: {avg_period:.4f} saniye (~{avg_period*1000:.1f} ms)")