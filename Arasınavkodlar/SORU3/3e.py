import numpy as np
import matplotlib.pyplot as plt

alpha = 0.95
beta = 0.95
omega = np.linspace(0, np.pi, 1000)

# G(ω) ve |G(ω)| hesaplama
G_omega = 1 / ((1 - alpha * np.exp(-1j * omega)) * (1 - beta * np.exp(-1j * omega)))
G_magnitude = np.abs(G_omega)

# G_tilde(ω) ve |G_tilde(ω)| hesaplama
G_tilde_omega = 1 / ((1 - alpha * np.exp(1j * omega)) * (1 - beta * np.exp(1j * omega)))
G_tilde_magnitude = np.abs(G_tilde_omega)

# Grafikler
plt.figure(figsize=(10, 5))
plt.plot(omega, G_magnitude, 'b-', label='|G(ω)|')
plt.plot(omega, G_tilde_magnitude, 'r--', label='|G_tilde(ω)|')
plt.title('|G(ω)| ve |G_tilde(ω)| Karşılaştırması (α=β=0.95)')
plt.xlabel('ω [rad]')
plt.ylabel('Genlik')
plt.xlim(0, np.pi)
plt.legend()
plt.grid(True)
plt.show()