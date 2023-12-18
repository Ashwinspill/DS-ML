import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer

ir = load_breast_cancer()
x = ir.data
y = ir.target

k = KMeans(n_clusters=3, random_state=42)
k.fit(x)
label = k.labels_
print(label)
centroids = k.cluster_centers_
print(centroids)


plt.scatter(x[:,0],x[:,1],c=label,cmap='viridis',marker='o')
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c='red')
plt.xlabel(ir.feature_names[0])
plt.ylabel(ir.feature_names[1])
plt.show()

