import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

NUM_CLUSTERS=3
colors = ['b','r', 'k']
markers = ['*', 'o', '+']

model = KeyedVectors.load_word2vec_format(os.path.join('..','Node2Vec','GoT_embeddings_5.emb'))
X = model[model.vocab]

def getElbow():
    # Determine Optimum Elbow
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = cluster.KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('Elbow_5')
    plt.show()

getElbow()

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print ("Cluster id labels for inputted data\n",labels)

#Applying PCA for dimensionality reduction of Node Features
pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)

plt.plot()
for i, l in enumerate(labels):
    plt.plot(pca_2d[i,0], pca_2d[i,1], color=colors[l], marker=markers[l],ls='None')
plt.title('K-means clustering on the GoT characters (PCA-reduced data)')
#plt.show()
plt.savefig('Plot_5')
