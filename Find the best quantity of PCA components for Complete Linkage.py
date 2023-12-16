from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score, completeness_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

scores = []
for components in range(1, X_pca.shape[1] + 1):
    X_pca_n = X_pca[:, :components]

    clustering = AgglomerativeClustering(linkage='complete')
    clustering.fit(X_pca_n)

    score = adjusted_rand_score(y, clustering.labels_)
    scores.append(score)
best_n_components = np.argmax(scores) + 1
print(f"Components with best result: {best_n_components}")

X_pca_best = X_pca[:, :best_n_components]

clustering_best = AgglomerativeClustering(linkage='complete')
clustering_best.fit(X_pca_best)

homogeneity = homogeneity_score(y, clustering_best.labels_)
completeness = completeness_score(y, clustering_best.labels_)

print(f"Homogenity:{homogeneity}\nCompleteness:{completeness}")