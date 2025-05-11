import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter

# Ses dosyasını oku ve mono'ya çevir
fs, data = wavfile.read("m18ae_40ms.wav")
if data.ndim > 1:
    data = data[:, 0]

# Kazanç (A) ve LP katsayıları (Önceki çıktınıza göre güncelleyin)
A = 0.03621  # Önceki kodunuzdaki gerçek değeri kullanın
lp_coeffs = np.array([
    1.0, -2.04321021, 1.80736675, -0.80876211, 0.13234092,
    -0.94948854, 1.82718531, -1.38800248, 0.04252149, 0.87636789,
    -0.29918086, -0.36054449, 0.42635847, 0.07957481, -0.34242757,
    0.10404746, -0.04071762, -0.02686721, -0.02489985, 0.03078988,
    0.05252582
])

# Periyodik impuls dizisi oluştur
duration = 0.04  # saniye
n_samples = int(fs * duration)
T0 = 0.008
T0_samples = int(T0 * fs)
impulse_train = np.zeros(n_samples)
impulse_train[::T0_samples] = 1

# Süzgeci uygula
filtered_signal = lfilter([A], lp_coeffs, impulse_train)

# Sinyali normalize et ve WAV formatına dönüştür
filtered_normalized = filtered_signal / np.max(np.abs(filtered_signal))  # [-1, 1] aralığı
wavfile.write("output_sound.wav", fs, (filtered_normalized * 32767).astype(np.int16))  # 16-bit PCM

# Sounddevice ile çal (Eğer yüklü değilse: pip install sounddevice)
try:
    import sounddevice as sd
    print("Ses çalınıyor...")
    sd.play(filtered_normalized, fs, blocking=True)
except ImportError:
    print("Sounddevice yüklü değil. Ses dosyası kaydedildi: 'output_sound.wav'")

# Grafikler
t_impulse = np.arange(n_samples) / fs
t_filtered = np.arange(len(filtered_signal)) / fs


# ... (Önceki kodun devamı)

# Sentezlenmiş sinyalin spektrumunu hesapla
def plot_log_spectrum(signal, fs, title):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    log_magnitude = 20 * np.log10(magnitude + 1e-12)  # dB cinsinden
    freq = np.fft.fftfreq(n, d=1/fs)[:n//2]
    log_magnitude = log_magnitude[:n//2]
    
    plt.figure(figsize=(10, 5))
    plt.plot(freq, log_magnitude, 'C2', linewidth=1.5)
    plt.xlim(0, fs/2)
    plt.xlabel('Frekans (Hz)', fontsize=12)
    plt.ylabel('Genlik (dB)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Sentezlenmiş sinyalin spektrumunu çiz
plot_log_spectrum(filtered_signal, fs, "Sentezlenmiş Sinyalin Logaritmik Genlik Spektrumu (H(ω))")

# (Opsiyonel) Orijinal sinyal ile karşılaştırma
# Orijinal sinyalin 40ms'lik penceresini al
original_segment = data[:n_samples]
hamming_window = np.hamming(n_samples)
windowed_original = original_segment * hamming_window

# İkisini aynı grafikte çiz
plt.figure(figsize=(12, 6))
freq_synth, spec_synth = plot_log_spectrum(filtered_signal, fs, return_data=True)
freq_orig, spec_orig = plot_log_spectrum(windowed_original, fs, return_data=True)

plt.plot(freq_orig, spec_orig, 'b', label='Orijinal Sinyal')
plt.plot(freq_synth, spec_synth, 'r--', label='Sentezlenmiş Sinyal (H(ω))')
plt.xlim(0, fs/2)
plt.xlabel('Frekans (Hz)', fontsize=12)
plt.ylabel('Genlik (dB)', fontsize=12)
plt.title('Spektral Karşılaştırma', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()