import numpy as np
import matplotlib.pyplot as plt

# Parametreler (istediğiniz gibi değiştirin)
a = 1    # Katsayı
d = 4    # Gecikme derecesi

# Sıfırların hesaplanması
zeros = np.roots([1] + [0]*(d-1) + [a])  # z^d + a = 0
poles = np.zeros(d)  # z^d = 0 → tüm kutuplar 0'da

# Kutup-sıfır diyagramı çizimi
plt.figure(figsize=(8, 8))
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Sıfırlar')
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Kutuplar')

# Birim çember çizimi
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.5)

# Eksenler ve başlık
plt.title(f'Kutup-Sıfır Diyagramı (a={a}, d={d})')
plt.xlabel('Reel')
plt.ylabel('İmajiner')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()