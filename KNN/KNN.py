import numpy as np

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def knn(k, train_data, test_data):
    results = []

    for test_f, _ in test_data:
        dists = [(train_f, t, dist(test_f, train_f)) for train_f, t in train_data]
        n_n = sorted(dists, key=lambda x: x[2])[:k]

        class_counts = {}
        for _, t, _ in n_n:
            class_counts[t] = class_counts.get(t, 0) + 1

        predicted_class = max(class_counts, key=class_counts.get)
        results.append(predicted_class)

    return results

storage = []
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

k = 5
results = knn(k, train_data, test_data)
with open('output.txt', 'w') as output_file:
    for result in results:
        output_file.write(str(result) + '\n')
