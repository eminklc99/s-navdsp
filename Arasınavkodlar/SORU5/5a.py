
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

# Ses dosyasını oku
sample_rate, data = wavfile.read('m18ae_40ms.wav')

# Stereo ise mono'ya dönüştür
if len(data.shape) > 1:
    data = data[:, 0]

# Zaman vektörünü oluştur
t = np.linspace(0, len(data)/sample_rate, len(data))

# Grafik çiz
plt.figure(figsize=(12, 4))
plt.plot(t, data)
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.title('m18ae_40ms.wav Ses Sinyali Zaman Grafiği')
plt.grid(True)
plt.show()