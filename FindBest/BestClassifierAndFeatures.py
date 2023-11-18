from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123456)

best_accuracy = 0.0
best_results = []

classifiers = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression()
}

feature_combinations = []
feature_names = iris.feature_names
num_features = len(feature_names)

for r in range(1, num_features + 1):
    feature_combinations.extend(combinations(range(num_features), r))

for classifier_name, classifier in classifiers.items():
    print(f"\nКлассификатор: {classifier_name}\n")

    for feature_indices in feature_combinations:
        X_train_subset = X_train[:, feature_indices]
        X_test_subset = X_test[:, feature_indices]
        classifier.fit(X_train_subset, y_train)
        y_pred = classifier.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy >= best_accuracy:
            if accuracy > best_accuracy:
                best_results = []
                best_accuracy = accuracy
            best_results.append({
                "Classifier": classifier_name,
                "Features": [feature_names[i] for i in feature_indices],
                "Accuracy": accuracy
            })

        print(f"Признаки: {', '.join([feature_names[i] for i in feature_indices])}, Точность: {accuracy}")

print("\nЛучшие результаты:")
for result in best_results:
    print(f"Классификатор: {result['Classifier']}")
    print(f"Признаки: {', '.join(result['Features'])}")
    print(f"Точность: {result['Accuracy']}")

""""""""""""""""""""""""""""""""""""""""""""""""
iris_df = pd.DataFrame(X, columns=iris.feature_names)
X_petal_width = iris_df[['petal width (cm)']]
X_train, X_test, y_train, y_test = train_test_split(X_petal_width, y, test_size=0.5, random_state=123456)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='red')
x_min, x_max = X_petal_width.min()-0.1, X_petal_width.max()+0.1
xx = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
yy = knn.predict(xx)
unique_classes = np.unique(yy)
separators = []

for i in range(len(unique_classes) - 1):
    class_1 = unique_classes[i]
    class_2 = unique_classes[i + 1]
    separator = (xx[yy == class_1].max() + xx[yy == class_2].min()) / 2
    separators.append(separator)
    plt.axvline(x=separator, color='black')

plt.xlabel('Petal Width (cm)')
plt.title('KNN Best classification')
plt.show()
""""""""""""""""""""""""""""""""""""""""""""""""
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

X = iris_df[['petal length (cm)', 'petal width (cm)']].values
y = iris_df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123456)

model = SVC()
model.fit(X_train, y_train)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.plasma)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.plasma, edgecolors='k')

plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('SVM с двумя признаками')
plt.show()