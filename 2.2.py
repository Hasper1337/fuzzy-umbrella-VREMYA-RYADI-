import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Определяем функции
functions = {
    'x²': lambda x: x**2,
    'x³': lambda x: x**3,
    'x': lambda x: x,
    'x⁻¹': lambda x: 1/x,
    '√x': lambda x: np.sqrt(x)
}

# Данные
x = np.linspace(0.1, 5, 500)
dx = x[1] - x[0]
func_values = {name: f(x) for name, f in functions.items()}

# Все пары
pairs = list(combinations(functions.keys(), 2))

# Вычисляем корреляции
lags = np.arange(-len(x)+1, len(x)) * dx
correlations = {}
for name1, name2 in pairs:
    corr = np.correlate(func_values[name1], func_values[name2], mode='full') * dx
    correlations[f"{name1} vs {name2}"] = corr

# Рисуем в одном цикле
fig, axes = plt.subplots(5, 2, figsize=(14, 20))
axes = axes.flatten()

for idx, ((name1, name2), ax) in enumerate(zip(pairs, axes)):
    corr = correlations[f"{name1} vs {name2}"]
    ax.plot(lags, corr, color='blue', linewidth=1.5)
    ax.set_title(f"{name1} ⋆ {name2}", fontsize=12)
    ax.set_ylabel("Корреляция")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.suptitle("Взаимная корреляция всех пар функций", fontsize=14)
plt.tight_layout()
plt.show()
