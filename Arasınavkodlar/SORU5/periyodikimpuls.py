import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ses dosyasını oku
fs, data = wavfile.read("m18ae_40ms.wav")

# Mono'ya çevir
if data.ndim > 1:
    data = data[:, 0]

# Örnekleme frekansı ve sinyal süresi
duration = 0.04  # saniye
n_samples = int(fs * duration)

# T0: daha önceki adımdan gelen periyot (örneğin yaklaşık 0.008 s)
T0 = 0.008
T0_samples = int(T0 * fs)

# Periyodik impuls dizisi oluştur
impulse_train = np.zeros(n_samples)
impulse_train[::T0_samples] = 1

# Zaman ekseni
t = np.arange(n_samples) / fs

# Çizdir
plt.figure(figsize=(10, 4))
plt.stem(t, impulse_train, basefmt=" ")
plt.xlabel("Zaman (s)")
plt.ylabel("İmpuls Değeri")
plt.title("Periyodik İmpuls Dizisi (T₀ ≈ {:.3f} s)".format(T0))
plt.grid(True)
plt.show()
