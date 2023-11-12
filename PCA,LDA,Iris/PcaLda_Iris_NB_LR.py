import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

def best_pca(classifier):
    accuracies = {}
    for n in range(1, X.shape[1] + 1):
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = classifier()
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[n] = accuracy

    best_n = max(accuracies, key=accuracies.get)
    return best_n, accuracies[best_n]

def best_lda(classifier):
    accuracies = {}
    max_components = min(len(np.unique(y)) - 1, X.shape[1])
    for n in range(1, max_components + 1):
        lda = LDA(n_components=n)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)

        clf = classifier()
        clf.fit(X_train_lda, y_train)
        y_pred = clf.predict(X_test_lda)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[n] = accuracy

    best_n = max(accuracies, key=accuracies.get)
    return best_n, accuracies[best_n]

nb_pca, nb_pca_accuracy = best_pca(GaussianNB)
lr_pca, lr_pca_accuracy = best_pca(LR)
nb_lda, nb_lda_accuracy = best_lda(GaussianNB)
lr_lda, lr_lda_accuracy = best_lda(LR)

print(f"PCA:")
print(f"Лучшее число компонент PCA для Байеса: {nb_pca}, Точность: {nb_pca_accuracy:.4f}")
print(f"Лучшее число компонент PCA для Лог. Регрессии: {lr_pca}, Точность: {lr_pca_accuracy:.4f}")
print("LDA:")
print(f"Лучшее число компонент LDA для Байеса: {nb_lda}, Точность: {nb_lda_accuracy:.4f}")
print(f"Лучшее число компонент LDA для Лог. Регрессии: {lr_lda}, Точность: {lr_lda_accuracy:.4f}")

def plot_regr_pca(n):
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    lr = LR()
    lr.fit(X_train_pca, y_train)

    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, s=20, edgecolor='k')
    plt.title(f"Логистическая Регрессия (PCA: n_components = {n}) на тестовой выборке")
    plt.show()
def plot_nb_pca(n):
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    nb = GaussianNB()
    nb.fit(X_train_pca, y_train)

    x_values = np.linspace(X_test_pca.min(), X_test_pca.max(), 300)
    predictions = nb.predict(x_values.reshape(-1, 1))

    change_points = x_values[np.where(np.diff(predictions) != 0)]

    plt.scatter(X_test_pca[:, 0], np.zeros(len(X_test_pca[:, 0])), c=y_test, alpha=1.0, edgecolor="black")
    for cp in change_points:
        plt.axvline(x=cp, color='red')
    plt.title(f"Байес (PCA: n_components = {n}) на тестовой выборке")
    plt.show()
def plot_nb_lda(n):
    lda = LDA(n_components=n)

    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    nb = GaussianNB()
    nb.fit(X_train_lda, y_train)

    x_values = np.linspace(X_test_lda.min(), X_test_lda.max(), 300)
    predictions = nb.predict(x_values.reshape(-1, 1))

    change_points = x_values[np.where(np.diff(predictions) != 0)]

    plt.scatter(X_test_lda[:, 0], np.zeros(len(X_test_lda[:, 0])), c=y_test, alpha=1.0, edgecolor="black")
    for cp in change_points:
        plt.axvline(x=cp, color='red')
    plt.title(f"Байес (LDA: n_components = {n}) на тестовой выборке")
    plt.show()
def plot_lr_lda(n):
    lda = LDA(n_components=n)
    lda.fit(X_train, y_train)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    lr = LR()
    lr.fit(X_train_lda, y_train)

    x_values = np.linspace(X_test_lda.min(), X_test_lda.max(), 300)
    predictions = lr.predict(x_values.reshape(-1, 1))

    change_points = x_values[np.where(np.diff(predictions) != 0)]

    plt.scatter(X_test_lda[:, 0], np.zeros(len(X_test_lda[:, 0])), c=y_test, alpha=1.0, edgecolor="black")
    for cp in change_points:
        plt.axvline(x=cp, color='red')
    plt.title(f"Логистическая Регрессия (LDA: n_components = {n}) на тестовой выборке")
    plt.show()

plot_regr_pca(lr_pca)
plot_nb_pca(nb_pca)
plot_nb_lda(nb_lda)
plot_lr_lda(lr_lda)