import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

# Ses dosyasını okuma ve Hamming penceresi ile pencereleme
fs, data = wavfile.read('m18ae_40ms.wav')
N = len(data)
hamming_window = np.hamming(N)
windowed_data = data * hamming_window

# Özilinti fonksiyonunu hesaplama
autocorrelation = np.correlate(windowed_data, windowed_data, mode='full')
tau = np.arange(-N + 1, N)  # τ = -639, ..., 639
zaman_tau = tau / fs  # τ'yu zamana çevirme (saniye)

# Zaman ekseninde çizdirme
plt.figure(figsize=(12, 5))
plt.plot(zaman_tau, autocorrelation)
plt.xlabel('Zaman (s)')
plt.ylabel('Özilinti Değeri')
plt.title('Pencerelenmiş Sinyalin Özilinti Fonksiyonu (Zaman Ekseninde)')
plt.grid()
plt.xlim(-0.04, 0.04)  # 40 ms'lik aralık
plt.show()

# Sadece pozitif τ değerlerini kullanarak tepe noktalarını bulma
autocorrelation_pozitif = autocorrelation[N-1:]  # τ ≥ 0 için
peaks, _ = find_peaks(autocorrelation_pozitif, distance=70, prominence=0.2*max(autocorrelation_pozitif))

# Tepe noktalarının zaman karşılıkları
peak_times = zaman_tau[N-1 + peaks]  # τ ≥ 0 olduğu için indeks ayarı
time_diff = np.diff(peak_times)
avg_period = np.mean(time_diff)

