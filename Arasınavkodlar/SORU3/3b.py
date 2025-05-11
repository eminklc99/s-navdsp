import numpy as np
import matplotlib.pyplot as plt

# Parametreler
alpha = 0.95
beta = 0.95
N = 401  # n = 0, 1, ..., 400

# Zaman Domeni: g[n] = (n+1)*(0.95)^n
n = np.arange(N)
g = (n + 1) * (alpha**n)  # α = β = 0.95 için basitleştirilmiş formül

# Frekans Domeni: |G(ω)| = 1 / |(1 - α e^{-iω})(1 - β e^{-iω})|
omega = np.linspace(0, np.pi, 1000)
G_omega = 1 / ( (1 - alpha * np.exp(-1j * omega)) * (1 - beta * np.exp(-1j * omega)) )
G_magnitude = np.abs(G_omega)

# Grafikler
plt.figure(figsize=(12, 6))

# Zaman Domeni Grafiği
plt.subplot(1, 2, 1)
plt.stem(n, g, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title('Zaman Domeni: g[n]')
plt.xlabel('n')
plt.ylabel('Genlik')
plt.xlim(0, 400)  # Uzun kuyruğu göstermek için sınırlandırıldı
plt.grid(True)

# Frekans Domeni Grafiği
plt.subplot(1, 2, 2)
plt.plot(omega, G_magnitude, 'r-')
plt.title('Frekans Cevabı: |G(ω)|')
plt.xlabel('ω [rad]')
plt.ylabel('Genlik')
plt.xlim(0, np.pi)
plt.grid(True)

plt.tight_layout()
plt.show()