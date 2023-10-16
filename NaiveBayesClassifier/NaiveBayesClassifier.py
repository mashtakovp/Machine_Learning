import numpy as np

storage = []
features = []

with open('input.txt', 'r') as input_file:
    for line in input_file.readlines()[1:]:
        lines = line.strip().split(',')
        features = list(map(float, lines[:4]))
        target = int(lines[4])
        storage.append((features, target))

train_data, test_data = [], []
for data in storage:
    if data[1] != -1:
        train_data.append(data)
    else:
        test_data.append(data)

# Сгруппировать данные по классам
class_sort = {}
for item in train_data:
    class_sort.setdefault(item[1], []).append(item[0])

# Вычислить среднее и стандартное отклонение для каждой функции
calc_dev = {target: {"mean_dev": np.mean(features, axis=0), "std_dev": np.std(features, axis=0)} for target, features in class_sort.items()}
# Функция для вычисления вероятности по закону нормального распределения
def calc_prob(x, mean_dev, std_dev):
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-((x - mean_dev) ** 2 / (2 * std_dev ** 2)))
# Классификация объектов из тестового набора
results = []
for features, _ in test_data:
    probs = {}
    print(calc_dev.items())
    for t, dev in calc_dev.items():
        class_prob = len(class_sort[t]) / len(train_data)
        prob = class_prob
        for i in range(len(features)):
            prob *= calc_prob(features[i], dev["mean_dev"][i], dev["std_dev"][i])
        probs[t] = prob
    predicted_class = max(probs, key=probs.get)
    results.append(predicted_class)


with open('output.txt', 'w') as output_file:
    for result in results:
        output_file.write(str(result) + '\n')
