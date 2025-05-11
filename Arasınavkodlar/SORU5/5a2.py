import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ses dosyasını oku
fs, data = wavfile.read("m18ae_40ms.wav")

# Eğer stereo ise, mono'ya çevir
if data.ndim > 1:
    data = data[:, 0]

# Zaman ekseni
t = np.arange(len(data)) / fs

# Ses sinyalini çiz
plt.figure(figsize=(10, 4))
plt.plot(t, data)
plt.xlabel("Zaman (s)")
plt.ylabel("Genlik")
plt.title("Konuşma Sinyali (Zaman Domeni)")
plt.grid()
plt.show()

from scipy.signal import correlate

# Oto-korelasyon
corr = correlate(data, data, mode='full')
corr = corr[len(corr)//2:]  # Pozitif gecikmeler

# İlk maksimumun konumunu bul (sıfırdan sonra)
d = np.diff(corr)
start = np.where(d > 0)[0][0]
peak = np.argmax(corr[start:]) + start

T0 = peak / fs
print(f"Tahmini periyot T₀ ≈ {T0:.4f} s")
