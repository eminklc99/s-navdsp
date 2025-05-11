import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

# 1. Ses Dosyasını Oku ve 16 kHz'e Örnekle
input_file = 'should.wav'
sample_rate_original, data = wavfile.read(input_file)

# Ses verisini tek kanala indir (mono) ve 16 kHz'e yeniden örnekle
target_sample_rate = 16000
if len(data.shape) > 1:
    data = data.mean(axis=1)  # Stereo -> Mono

# Yeniden örnekleme
data_resampled = signal.resample(data, int(len(data) * target_sample_rate / sample_rate_original))
data_resampled = data_resampled.astype(np.float32)
data_resampled /= np.max(np.abs(data_resampled))  # Normalize et

# 2. Tüm Echo Kombinasyonlarını Üret
delays_ms = [1, 10, 100, 1000]
alphas = [0.1, 0.5, 0.9, 1.0]
outputs = []

for delay_ms in delays_ms:
    for alpha in alphas:
        D = int(delay_ms * target_sample_rate / 1000)
        output = np.zeros(len(data_resampled) + D)
        output[:len(data_resampled)] = data_resampled
        output[D:D+len(data_resampled)] += alpha * data_resampled
        output = output / np.max(np.abs(output))  # Clipping önleme
        outputs.append((delay_ms, alpha, output))

# 3. Çıktıları Kaydet
os.makedirs('echo_outputs', exist_ok=True)
for i, (delay_ms, alpha, output) in enumerate(outputs):
    output_int16 = (output * 32767).astype(np.int16)
    wavfile.write(f'echo_outputs/echo_d{delay_ms}ms_a{alpha}.wav', target_sample_rate, output_int16)

# 4. Karşılaştırmalı Analiz Grafikleri
plt.figure(figsize=(20, 20))

# Zaman Domain Karşılaştırması
for i, (delay_ms, alpha, output) in enumerate(outputs):
    plt.subplot(4, 4, i+1)
    plt.plot(output[:5000])  # İlk 500 örneği göster
    plt.title(f'D={delay_ms}ms, α={alpha}')
    plt.xlabel('Örnek')
    plt.ylabel('Genlik')
    plt.ylim(-1.1, 1.1)
    plt.grid(True)

plt.tight_layout()
plt.savefig('echo_comparison_time.png')
plt.show()

# Frekans Domain Karşılaştırması
plt.figure(figsize=(20, 20))
for i, (delay_ms, alpha, output) in enumerate(outputs):
    plt.subplot(4, 4, i+1)
    fft = np.fft.rfft(output)
    freq = np.fft.rfftfreq(len(output), 1/target_sample_rate)
    plt.semilogx(freq, 20*np.log10(np.abs(fft)+1e-9))
    plt.title(f'D={delay_ms}ms, α={alpha}')
    plt.xlabel('Frekans (Hz)')
    plt.ylabel('Genlik (dB)')
    plt.xlim(20, 8000)
    plt.ylim(-60, 0)
    plt.grid(True)

plt.tight_layout()
plt.savefig('echo_comparison_freq.png')
plt.show()

print("Tüm işlemler tamamlandı! Çıktılar 'echo_outputs' klasöründe.")