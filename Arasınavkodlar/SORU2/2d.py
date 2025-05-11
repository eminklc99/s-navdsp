

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parametreler
formant_freq = 500   # Formant frekansı (Hz)
sampling_freq = 8000 # Örnekleme frekansı (Bu değer varsayılmıştır!)
theta = 2 * np.pi * formant_freq / sampling_freq  # Normalize açısal frekans

r_values = [0.2, 0.9]  # Test edilecek r değerleri
colors = ['blue', 'red']  # Grafik renkleri

# Genlik cevaplarını çiz
plt.figure(figsize=(12, 6))

for r, color in zip(r_values, colors):
    # Transfer fonksiyonu katsayıları
    b = [1]  # Pay (numerator)
    a = [1, -2 * r * np.cos(theta), r**2]  # Payda (denominator)
    
    # Frekans cevabını hesapla
    w, h = signal.freqz(b, a)
    frekans = (w * sampling_freq) / (2 * np.pi)  # Hz'e çevir
    
    # Genlik cevabını çiz (dB cinsinden)
    plt.plot(frekans, 20 * np.log10(np.abs(h)), 
             color=color, 
             label=f'r = {r}', 
             linewidth=2)

# Grafik ayarları
plt.title('Resonatör Genlik Cevabı (Formant Frekansı = 500 Hz)', fontsize=14)
plt.xlabel('Frekans (Hz)', fontsize=12)
plt.ylabel('Genlik (dB)', fontsize=12)
plt.xlim(0, 2000)  # Sadece 0-2000 Hz arası göster
plt.axvline(formant_freq, color='gray', linestyle='--', label='Formant (500 Hz)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()