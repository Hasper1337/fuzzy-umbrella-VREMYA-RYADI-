import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала
A0 = 2
f0 = 1.0
w0 = 2 * np.pi * f0
phi0 = np.pi/4

# Параметры дискретизации
N = 2048        # Количество точек
dt = 0.005      # Шаг по времени
T = N * dt      # Общее время наблюдения
t = np.linspace(0, T, N, endpoint=False)

f = A0 * np.sin(w0 * t + phi0)

freq = np.fft.fftfreq(N, dt)

def manual_fft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

F_manual = manual_fft(f)

F_manual_scaled = F_manual * dt

def manual_ifft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

f_reconstructed = manual_ifft(F_manual)


F_numpy = np.fft.fft(f)
f_numpy_reconstructed = np.fft.ifft(F_numpy)

F_numpy_scaled = F_numpy * dt

freq_shifted = np.fft.fftshift(freq)
F_manual_shifted = np.fft.fftshift(F_manual_scaled)
F_numpy_shifted = np.fft.fftshift(F_numpy_scaled)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(t, f, '-b', linewidth=1.5)
axes[0].set_title('Исходный сигнал', fontsize=12)
axes[0].set_xlabel('Время (с)')
axes[0].set_ylabel('Амплитуда')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 2])

axes[1].plot(t, np.real(f_reconstructed), ':g', linewidth=4, label='Ручная реализация')
axes[1].plot(t, np.real(f_numpy_reconstructed), '--r', linewidth=2, label='NumPy iFFT')
axes[1].plot(t, f, 'b-', linewidth=1, label='Исходный')
axes[1].set_title('Обратное преобразование Фурье', fontsize=12)
axes[1].set_xlabel('Время (с)')
axes[1].set_ylabel('Амплитуда')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 2])

axes[2].plot(freq_shifted, np.abs(F_numpy_shifted), '-r', linewidth=1.5, alpha=0.7, label='NumPy FFT')
axes[2].plot(freq_shifted, np.abs(F_manual_shifted), 'Xg', linewidth=0.7,alpha=0.7, label='Ручная реализация')
axes[2].set_title('Прямое преобразование Фурье', fontsize=12)
axes[2].set_xlabel('Частота (Гц)')
axes[2].set_ylabel('|F(y)|')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([-5, 5])

plt.tight_layout()
plt.show()