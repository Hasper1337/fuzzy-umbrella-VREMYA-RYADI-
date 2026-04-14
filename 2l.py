import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

N = 1000  # количество точек
x_start, x_end = -0.5, 1.5

# cетка
x = np.linspace(x_start, x_end, N)
dx = x[1] - x[0] # шаг дискретизации

# ступенчатый сигнал
def rect_signal(t):
    return np.where((t > 0) & (t < 1), 1.0, 0.0)

f = rect_signal(x)

# Реализовать по формуле
def manual_convolution(f, g, dx):

    n = len(f)
    result = np.zeros(2 * n - 1)

    for i in range(2 * n - 1):

        for k in range(n):

            j = i - k
            if 0 <= j < n:

                result[i] += f[k] * g[j] * dx

    return result

conv_scipy = convolve(f, f, mode='full') * dx

x_conv = np.linspace(2*x_start, 2*x_end, len(conv_scipy))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# исходный сигнал
axes[0].plot(x, f, 'b-', linewidth=2.5)
axes[0].set_title('Исходный ступенчатый сигнал f(x)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].grid(True, alpha=0.3)

# свёртка
conv_manual = manual_convolution(f, f, dx)
x_manual = np.linspace(2*x_start, 2*x_end, len(conv_manual))

axes[1].plot(x_manual, conv_manual, '-g', linewidth=2.5, label='(f * g)(x) - manual_convolution')
axes[1].plot(x_conv, conv_scipy, ':r', linewidth=2.5, alpha=0.5,label='(f * g)(x) - scipy.convolve')
axes[1].set_title('Реализация по формуле', fontsize=12, fontweight='bold')
axes[1].set_xlabel('x')
axes[1].set_ylabel('(f * g)(x)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()