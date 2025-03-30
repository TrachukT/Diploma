import matplotlib.pyplot as plt
import numpy as np

# Дані
sizes = ["Малий", "Середній", "Великий"]
sequential = [6.5743, 5.6716, 5.9626]
parallel_2 = [14.7605, 11.8612, 10.6746]
parallel_4 = [1.9848, 0.2458, 0.5886]
parallel_8 = [0.9084, 0.5784, 0.3066]

# Побудова графіка
x = np.arange(len(sizes))  # Позиції по осі X
width = 0.2  # Ширина стовпчиків

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 1.5*width, sequential, width, label="Послідовне")
ax.bar(x - 0.5*width, parallel_2, width, label="2 потоки")
ax.bar(x + 0.5*width, parallel_4, width, label="4 потоки")
ax.bar(x + 1.5*width, parallel_8, width, label="8 потоків")

# Налаштування осей
ax.set_xlabel("Розмір графа")
ax.set_ylabel("Час сортування (мс)")
ax.set_title("Порівняння часу сортування")
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()

# Відображення графіка
plt.show()