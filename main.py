
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

df = pd.read_excel("data.xlsx")

data = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy()

length = len(data) 
mean = np.mean(data) #Математическое ожидание
variance = np.var(data, ddof=1) #Выборочная дисперсия
std = np.std(data, ddof=1) #Среднеквадратическое отклонение
cv = std / mean if mean != 0 else float("nan") #Коэффициент вариации

print("\nКоличество измерений:", length)
print("Математическое ожидание:", mean)
print("Выборочная дисперсия:", variance)
print("Среднеквадратическое отклонение:", std)
print("Коэффициент вариации:", cv)

def calculate_confidence_interval(length, mean, std, confidence):
    intervals = {}
    for level in confidence:
        alpha = 1 - level
        t_value = sp.stats.t.ppf(1 - alpha / 2, df=length - 1)
        margin_of_error = t_value * (std / np.sqrt(length))

        intervals[level] = {
            "lower": mean - margin_of_error,
            "upper": mean + margin_of_error,
            "margin": margin_of_error,
        }

    return intervals

intervals = calculate_confidence_interval(
    length, mean, std, [0.90, 0.95, 0.99]
)

for level, interval in intervals.items():
    print(
        f"Доверительный интервал {level * 100:.0f}%: "
        f"{mean:.4f} ± {interval['margin']:.4f}"
    )

def relative_difference(value1, value2):
    if value2 == 0:
        return float("inf")  # Avoid division by zero
    return abs(value1 - value2) / abs(value2) * 100


for n in [10, 20, 50, 100, 200, 300]:
    sample = data[:n]

    sample_mean = np.mean(sample)
    sample_variance = np.var(sample, ddof=1)
    sample_std = np.std(sample, ddof=1)
    sample_cv = sample_std / sample_mean if sample_mean != 0 else float("nan")

    print(f"\nПоказатели для выборки из {n} значений:")
    print(f"Математическое ожидание: {sample_mean:.4f}")
    print(f"Выборочная дисперсия: {sample_variance:.4f}")
    print(f"Среднеквадратическое отклонение: {sample_std:.4f}")
    print(f"Коэффициент вариации: {sample_cv:.4f}")

    print(f"\nОтносительные отклонения для выборки из {n} значений:")
    print(f"Математическое ожидание: {relative_difference(sample_mean, mean):.2f}%")
    print(f"Выборочная дисперсия: {relative_difference(sample_variance, variance):.2f}%")
    print(f"Среднеквадратическое отклонение: {relative_difference(sample_std, std):.2f}%")
    print(f"Коэффициент вариации: {relative_difference(sample_cv, cv):.2f}%")

    sample_intervals = calculate_confidence_interval(
        len(sample), sample_mean, sample_std, [0.90, 0.95, 0.99]
    )

    for level, interval in sample_intervals.items():
        deviation = relative_difference(
            interval["margin"],
            intervals[level]["margin"]
        )

        print(
            f"Доверительный интервал {level * 100:.0f}%: "
            f"{sample_mean:.4f} ± {interval['margin']:.4f}; "
            f"отклонение полуширины: {deviation:.2f}%"
        )

numbers_test1 = np.arange(1, len(data) + 1)

def diagram(data, numbers, title):
    plt.figure(figsize=(12, 5))
    plt.plot(numbers, data, marker=".", markersize=4, linewidth=0.8)

    plt.title("График "+title)
    plt.xlabel("Номер измерения")
    plt.ylabel("Значение")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

diagram(data, numbers_test1, "исходной последовательности")
"""
Последовательность имеет колебательный характер без выраженной тенденции к росту или снижению.
По графику явная периодичность не обнаружена.
Для проверки зависимости между значениями требуется автокорреляционный анализ
"""


#Автокорреляционный анализ

def autocorrelation(data, lag):
    first = data[:-lag]
    second = data[lag:]

    return np.corrcoef(first, second)[0, 1]


lags = np.arange(1, 11)

correlations = np.array([
    autocorrelation(data, lag)
    for lag in lags
])

print("\nАвтокорреляция исходной последовательности:")

for lag, correlation in zip(lags, correlations):
    print(f"Сдвиг {lag}: {correlation:.4f}")

limit = 1.96 / np.sqrt(len(data))

plt.figure(figsize=(10, 5))
plt.stem(lags, correlations)

plt.axhline(
    limit,
    color="red",
    linestyle="--",
    label="Приблизительные 95%-е границы"
)
plt.axhline(-limit, color="red", linestyle="--")
plt.axhline(0, color="black", linewidth=0.8)

plt.xticks(lags)
plt.xlabel("Сдвиг (лаг)")
plt.ylabel("Коэффициент автокорреляции")
plt.title("Автокорреляция исходной последовательности")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

"""
На лагах от 1 до 10 большинство коэффициентов автокорреляции находится внутри приблизительных 95%-х границ +-0,113
На лаге 8 наблюдается отрицательный коэффициент около -0,138, выходящий за нижнюю границу. 
Устойчивого периодического рисунка автокорреляции на проверенных лагах не обнаружено.
"""

# Гистограмма распределения частот

plt.figure(figsize=(10, 5))

counts, edges, _ = plt.hist(
    data,
    bins=10,
    edgecolor="black",
    color="skyblue"
)

plt.title("Гистограмма распределения частот исходной последовательности")
plt.xlabel("Значение")
plt.ylabel("Частота — количество измерений")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Параметры распределения Эрланга
k = 3
scale = mean / k

# Точки для построения теоретической плотности
x = np.linspace(0, data.max() * 1.1, 1000)
density = sp.stats.erlang.pdf(x, a=k, scale=scale)

plt.figure(figsize=(10, 5))

# Нормированная гистограмма исходных данных
plt.hist(
    data,
    bins=10,
    density=True,
    color="skyblue",
    edgecolor="black",
    alpha=0.6,
    label="Исходные данные"
)

# Плотность распределения Эрланга
plt.plot(
    x,
    density,
    color="red",
    linewidth=2,
    label=f"Эрланг: k={k}, scale={scale:.2f}"
)

plt.title("Аппроксимация исходных данных распределением Эрланга")
plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Генератор Эрланга третьего порядка
k = 3
rng = np.random.default_rng(42)

generated_data = rng.exponential(
    scale=mean / k,
    size=(len(data), k)
).sum(axis=1)

# Два столбца: исходные и сгенерированные значения
result = pd.DataFrame({
    "Исходная последовательность": data,
    "Эрланг k=3": generated_data
})

result.to_excel("generated.xlsx", index=False)

print("\nСоздан файл generated.xlsx")
print(f"Сгенерировано значений: {len(generated_data)}")
print(f"Среднее: {np.mean(generated_data):.4f}")
print(f"Дисперсия: {np.var(generated_data, ddof=1):.4f}")

for n in [10, 20, 50, 100, 200, 300]:
    sample = generated_data[:n]
    reference = data[:n]

    reference_mean = np.mean(reference)
    reference_variance = np.var(reference, ddof=1)
    reference_std = np.std(reference, ddof=1)
    reference_cv = (
        reference_std / reference_mean
        if reference_mean != 0 else float("nan")
    )

    reference_intervals = calculate_confidence_interval(
        len(reference),
        reference_mean,
        reference_std,
        [0.90, 0.95, 0.99]
    )
    
    sample_mean = np.mean(sample)
    sample_variance = np.var(sample, ddof=1)
    sample_std = np.std(sample, ddof=1)
    sample_cv = sample_std / sample_mean if sample_mean != 0 else float("nan")

    print(f"\nПоказатели для выборки из {n} значений:")
    print(f"Математическое ожидание: {sample_mean:.4f}")
    print(f"Выборочная дисперсия: {sample_variance:.4f}")
    print(f"Среднеквадратическое отклонение: {sample_std:.4f}")
    print(f"Коэффициент вариации: {sample_cv:.4f}")

    print(f"\nОтносительные отклонения для выборки из {n} значений:")
    print(f"Математическое ожидание: {relative_difference(sample_mean, reference_mean):.2f}%")
    print(f"Выборочная дисперсия: {relative_difference(sample_variance, reference_variance):.2f}%")
    print(f"Среднеквадратическое отклонение: {relative_difference(sample_std, reference_std):.2f}%")
    print(f"Коэффициент вариации: {relative_difference(sample_cv, reference_cv):.2f}%")

    sample_intervals = calculate_confidence_interval(
        len(sample), sample_mean, sample_std, [0.90, 0.95, 0.99]
    )

    for level, interval in sample_intervals.items():
        deviation = relative_difference(
            interval["margin"],
            reference_intervals[level]["margin"]
        )

        print(
            f"Доверительный интервал {level * 100:.0f}%: "
            f"{sample_mean:.4f} ± {interval['margin']:.4f}; "
            f"отклонение полуширины: {deviation:.2f}%"
        )

lags = np.arange(1, 11)

correlations = np.array([
    autocorrelation(generated_data, lag)
    for lag in lags
])

print("\nАвтокорреляция сгенерированной последовательности:")

for lag, correlation in zip(lags, correlations):
    print(f"Сдвиг {lag}: {correlation:.4f}")

limit = 1.96 / np.sqrt(len(generated_data))

plt.figure(figsize=(10, 5))
plt.stem(lags, correlations)

plt.axhline(
    limit,
    color="red",
    linestyle="--",
    label="Приблизительные 95%-е границы"
)
plt.axhline(-limit, color="red", linestyle="--")
plt.axhline(0, color="black", linewidth=0.8)

plt.xticks(lags)
plt.xlabel("Сдвиг (лаг)")
plt.ylabel("Коэффициент автокорреляции")
plt.title("Автокорреляция сгенерированной последовательности")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# Корреляция между исходной и сгенерированной последовательностями

generated_correlation = np.corrcoef(data, generated_data)[0, 1]
print(f"Корреляция между последовательностями: {generated_correlation:.4f}")

# Сравнение графиков значений
numbers = np.arange(1, len(data) + 1)

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 8),
    sharex=True,
    sharey=True
)

axes[0].plot(
    numbers, data,
    color="steelblue",
    linewidth=0.8
)
axes[0].set_title("Исходная последовательность")
axes[0].set_ylabel("Значение")

axes[1].plot(
    numbers, generated_data,
    color="darkorange",
    linewidth=0.8
)
axes[1].set_title("Сгенерированная последовательность")
axes[1].set_xlabel("Номер измерения")
axes[1].set_ylabel("Значение")

for ax in axes:
    ax.grid(True, alpha=0.3)

fig.suptitle("Сравнение исходной и сгенерированной последовательностей")
plt.tight_layout()
plt.show()

# Общие границы 10 интервалов для обеих последовательностей
combined_data = np.concatenate([data, generated_data])
common_edges = np.histogram_bin_edges(combined_data, bins=10)

plt.figure(figsize=(12, 5))

plt.hist(
    [data, generated_data],
    bins=common_edges,
    color=["steelblue", "darkorange"],
    edgecolor="black",
    label=["Исходная последовательность", "Сгенерированная последовательность"]
)

plt.title("Сравнение гистограмм распределения частот")
plt.xlabel("Значение")
plt.ylabel("Частота — количество измерений")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\nСравнение коэффициентов автокорреляции:")
print(f"{'Лаг':>3} {'Исходная':>12} {'Сгенерированная':>16} {'Отклонение, %':>16}")

for lag in range(1, 11):
    original_corr = autocorrelation(data, lag)
    generated_corr = autocorrelation(generated_data, lag)

    deviation = relative_difference(generated_corr, original_corr)

    print(
        f"{lag:>3} "
        f"{original_corr:>12.4f} "
        f"{generated_corr:>16.4f} "
        f"{deviation:>16.2f}"
    )