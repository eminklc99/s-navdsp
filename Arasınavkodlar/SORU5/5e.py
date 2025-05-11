import numpy as np
from scipy.io import wavfile
import librosa

# Ses dosyasını oku ve mono'ya çevir
sample_rate, data = wavfile.read('m18ae_40ms.wav')
if len(data.shape) > 1:
    data = data[:, 0]

# 40 ms'lik Hamming penceresi uygula
window_length = int(0.04 * sample_rate)
signal_segment = data[:window_length]  # İlk 40 ms'lik kısım
hamming_window = np.hamming(window_length)
windowed_signal = signal_segment * hamming_window

# Sinyali normalize et (Librosa ile tutarlılık için)
windowed_signal = windowed_signal / np.max(np.abs(windowed_signal))

# ---------------------
# 1. ÖZİLİNTİ HESAPLAMA 
# ---------------------
p = 20
n = len(windowed_signal)
r = np.zeros(p + 1)

# **Nötr (unbiased) özilinti** kullanılmalı
for k in range(p + 1):
    r[k] = np.sum(windowed_signal[k:] * windowed_signal[:n - k]) / (n - k)  # Düzeltme!

# ---------------------------
# 2. A KAZANCI HESAPLAMA 
# ---------------------------
# Verilen LP katsayıları (alpha_custom)
alpha_custom = np.array([
    2.04329167, -1.80745716, 0.80866572, -0.13175562, 0.9482427,
    -1.82588242, 1.38733435, -0.04238493, -0.87668338, 0.29998738,
    0.35980254, -0.42623509, -0.07922955, 0.34230059, -0.10438987,
    0.04121287, 0.02656084, 0.0249514, -0.03081125, -0.05249107
])

# Hata enerjisi formülü: e[p] = r[0] - Σ αk * r[k] (EKSİ İŞARETLİ!)
e_p_custom = r[0] - np.sum(alpha_custom * r[1:])  # Düzeltme!
A_custom = np.sqrt(e_p_custom)
print(f"Kendi Yöntemle A Kazancı: {A_custom:.3f}")

# ---------------------------
# 3. LIBROSA İLE KARŞILAŞTIRMA
# ---------------------------
alpha_librosa = librosa.lpc(windowed_signal, order=20)

# Librosa'nın katsayıları ile A'yı hesapla (aynı formül)
e_p_librosa = r[0] - np.sum(alpha_librosa[1:] * r[1:])  # Düzeltme!
A_librosa = np.sqrt(e_p_librosa)
print(f"Librosa ile A Kazancı: {A_librosa:.3f}")

# ---------------------------
# 4. İŞARET TUTARLILIĞI KONTROLÜ
# ---------------------------
# Librosa'nın katsayıları işaret olarak ters olabilir
alpha_librosa_normalized = -alpha_librosa[1:]  # İşaretleri ters çevir

# Farkı hesapla
fark = np.abs(A_custom - A_librosa)
print(f"\nFark: {fark:.3f}")

if fark < 1e-3:
    print("Sonuçlar tutarlı!")
else:
    print("Uyarı: Hata devam ediyor. Muhtemel nedenler:")
    print("- LP katsayıları yanlış hesaplanmış (Levinson-Durbin algoritmasında hata)")
    print("- Sinyal penceresi doğru uygulanmamış")