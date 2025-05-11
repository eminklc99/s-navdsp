import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Dosyayı okuma
fs, data = wavfile.read('m18ae_40ms.wav')

N = len(data)  # Örnek sayısı = 16000 Hz * 0.04 s = 640
zaman = np.arange(N) / fs  # Zaman vektörü (saniye)

plt.figure(figsize=(10, 4))
plt.plot(zaman, data)
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.title('m18ae_40ms.wav Sinyali')
plt.grid()
plt.show()