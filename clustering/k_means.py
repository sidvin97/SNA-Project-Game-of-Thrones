import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt

NUM_CLUSTERS=3

model = KeyedVectors.load_word2vec_format(os.path.join('..','Node2Vec','GoT_embeddings.emb'))
X = model[model.vocab]

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
 
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))

'''
plt.plot()
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(model[i], model[i], color=colors[l], marker=markers[l],ls='None')
    #plt.xlim([0, 10])
    #plt.ylim([0, 10])

plt.show()
'''
