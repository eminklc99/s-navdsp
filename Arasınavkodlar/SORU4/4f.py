# Örnek Python kodu (sonuç için)
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ses dosyasını oku
fs, data = wavfile.read('m18oo_40ms.wav')

# Zaman eksenini oluştur
time = np.arange(len(data)) / fs

# Grafik çiz
plt.plot(time, data)
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.title('m18oo_40ms.wav Sinyali')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ses dosyasını oku
fs, data = wavfile.read('m18oo_40ms.wav')

# Hamming penceresini oluştur
pencere = np.hamming(len(data))

# Sinyali pencerele
pencere_sinyal = data * pencere

# Zaman ekseni
time = np.arange(len(data)) / fs

# Grafik çiz
plt.figure(figsize=(10, 4))
plt.plot(time, pencere_sinyal)
plt.xlabel('Zaman (s)')
plt.ylabel('Genlik')
plt.title('Hamming Pencerelenmiş Sinyal (40 ms)')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ses dosyasını oku
fs, data = wavfile.read('m18oo_40ms.wav')

# Hamming penceresini uygula
pencere = np.hamming(len(data))
pencere_sinyal = data * pencere

# Özilinti fonksiyonunu hesapla
autocorr = np.correlate(pencere_sinyal, pencere_sinyal, mode='full')

# Lag değerlerini oluştur
lags = np.arange(-len(pencere_sinyal) + 1, len(pencere_sinyal))

# Grafik çiz
plt.figure(figsize=(12, 6))
plt.plot(lags, autocorr)
plt.xlabel('τ (Lag)')
plt.ylabel('Özilinti')
plt.title('Pencerelenmiş Sinyalin Özilinti Fonksiyonu')
plt.grid(True)
plt.xlim(-700, 700)  # Yakın bölgeyi vurgula
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Ses dosyasını oku
fs, data = wavfile.read('m18oo_40ms.wav')

# Hamming penceresini uygula
pencere = np.hamming(len(data))
pencere_sinyal = data * pencere

# Özilinti fonksiyonunu hesapla
autocorr = np.correlate(pencere_sinyal, pencere_sinyal, mode='full')

# Zaman eksenini oluştur (τ'yu saniyeye çevir)
lags = np.arange(-len(pencere_sinyal) + 1, len(pencere_sinyal)) / fs

# Grafik çiz
plt.figure(figsize=(12, 6))
plt.plot(lags, autocorr)
plt.xlabel('Zaman (s)')
plt.ylabel('Özilinti')
plt.title('Özilinti Fonksiyonu (Zaman Ekseninde)')
plt.grid(True)
plt.xlim(-0.04, 0.04)  # 40 ms'lik pencereyi vurgula
plt.show()

# Periyot hesapla (örnek: ilk iki tepe arası τ=80 örnek)
T_autocorr = 80 / fs  # 80 örnek → 0.005 s
print(f"Özilinti Fonksiyonundan Hesaplanan Periyot: {T_autocorr} s")

# Önceki değerlerle karşılaştır
print(f"İlk Adım (Ham Sinyal): T ≈ 0.005 s")
print(f"Üçüncü Adım (Özilinti): T ≈ {T_autocorr} s")