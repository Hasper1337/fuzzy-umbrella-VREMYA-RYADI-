import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ['Ничего', 'Агрессивная', 'Белый шум', 'Классическая', 'Ритмичная']

df = pd.read_csv("lab3_data1.txt", sep='\t', names=columns)

fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(15, 10))
fig.suptitle('Временные ряды сердцебиения при прослушивании разной музыки', fontsize=16)

dt = 750 # шаг дискретизации
max_time = 10000 # время в мс

for i, col in enumerate(columns):
    intervals = df[col].dropna().values

    beat_times = np.cumsum(intervals).astype(int)
    beat_times = beat_times[beat_times <= max_time]

    step_dis = (max_time // dt) + 1

    time_series = np.zeros(step_dis, dtype=int)
    indeces = beat_times // dt
    time_series[indeces] = 1
    time_axis = np.arange(step_dis) * dt
    axes[i].axhline(0, color='#1f77b4', linewidth=1.5)
    axes[i].vlines(beat_times, ymin=0, ymax=1, color='#1f77b4', linewidth=1.5)

    axes[i].set_title(col)


plt.xlabel('Время (миллисекунды)')

plt.tight_layout()
plt.show()