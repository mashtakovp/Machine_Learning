import numpy as np

rate = 1/100
epochs = 200

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


x_train, y_train = [], []

for i in train_data:
    x_train.append(i[0])
    y_train.append(i[1])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []

for pair in test_data:
    x_test.append(pair[0])
x_test = np.array(x_test)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def train(x, y, rate, epochs):
    w = np.zeros(len(x.T))
    b = 0

    for epoch in range(epochs):
        a_train = sigmoid(x@w + b)
        dw = x.T@(a_train - y) / len(x)
        db = np.sum(a_train - y) / len(x)
        w = w - rate * dw
        b = b - rate * db

    return w, b

w, b = train(x_train, y_train, rate, epochs)

a_test = sigmoid(x_test@w + b)
results = []
for i in range(len(a_test)):
    if a_test[i] >= 0.5:
        results.append(1)
    else:
        results.append(0)

with open('output.txt', 'w') as output_file:
    for result in results:
        output_file.write(str(result) + '\n')
