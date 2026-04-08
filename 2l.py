
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

N = 1000  # количество точек
x_start, x_end = -0.5, 2.5

# cетка
x = np.linspace(x_start, x_end, N)
dx = x[1] - x[0]

# ступенчатый сигнал
def rect_signal(t):
    return np.where((t > 0) & (t < 1), 1.0, 0.0)

f = rect_signal(x)

# Реализовать по формуле
def analytical_convolution_rect(t):
    result = np.zeros_like(t, dtype=float)
    mask1 = (t >= 0) & (t <= 1)
    mask2 = (t > 1) & (t <= 2)

    result[mask1] = t[mask1]
    result[mask2] = 2 - t[mask2]

    return result

conv_scipy = convolve(f, f, mode='full') * dx  # Умножаем на dx для интегрирования

# Ось для результата свёртки: сумма аргументов
# Если x от a до b, то x_conv от 2a до 2b
x_conv = np.linspace(2*x_start, 2*x_end, len(conv_scipy))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# исходный сигнал
ax1 = axes[0]

ax1.plot(x, f, 'b-', linewidth=2.5, label='f(x) = rect(x; 0,1)')
ax1.set_title('Исходный ступенчатый сигнал f(x)', fontsize=12, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.1, 1.2)
ax1.grid(True, alpha=0.3)
ax1.legend()

# свёртка по формуле
ax2 = axes[1]
x_analytical = np.linspace(-0.5, 2.5, 1000)
conv_analytical = analytical_convolution_rect(x_analytical)

ax2.plot(x_analytical, conv_analytical, 'g-', linewidth=2.5, label='(f * f)(x) - аналитически')
ax2.set_title('Часть 1: Реализация по формуле (аналитически)', fontsize=12, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('(f * f)(x)')
ax2.set_xlim(-0.5, 2.5)
ax2.set_ylim(-0.1, 1.2)
ax2.grid(True, alpha=0.3)
ax2.legend()

# библиотечная свёртка
ax3 = axes[2]

mask_display = (x_conv >= -0.5) & (x_conv <= 2.5)

ax3.plot(x_conv[mask_display], conv_scipy[mask_display], 'r-', linewidth=2.5, label='(f * f)(x) - scipy.convolve')
ax3.set_title('Часть 2: Реализация с scipy.signal.convolve', fontsize=12, fontweight='bold')
ax3.set_xlabel('x')
ax3.set_ylabel('(f * f)(x)')
ax3.set_xlim(-0.5, 2.5)
ax3.set_ylim(-0.1, 1.2)
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()

# Проверка максимума
print(f"Аналитический максимум: {np.max(conv_analytical):.4f} (должен быть 1.0)")
print(f"Scipy максимум: {np.max(conv_scipy):.4f}")
print(f"Ось x_conv от {x_conv[0]:.2f} до {x_conv[-1]:.2f}")
