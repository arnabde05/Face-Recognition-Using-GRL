import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# set the paths for train and test embeddings
train_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\training\\90"
test_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\testing\\75"

# read the embeddings for train and test images and store their respective IDs
train_embeddings = []
train_ids = []
for filename in os.listdir(train_path):
    if filename.endswith(".emb"):
        emb_path = os.path.join(train_path, filename)
        train_embeddings.append(np.loadtxt(emb_path))
        train_id = filename.split("_")[0]
        train_ids.append(train_id)
train_embeddings = np.array(train_embeddings)

test_embeddings = []
test_ids = []
for filename in os.listdir(test_path):
    if filename.endswith(".emb"):
        emb_path = os.path.join(test_path, filename)
        test_embeddings.append(np.loadtxt(emb_path))
        test_id = filename.split("_")[0]
        test_ids.append(test_id)
test_embeddings = np.array(test_embeddings)

# cluster the training embeddings
n_clusters = len(np.unique(train_ids))
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
kmeans.fit(train_embeddings)
train_labels = kmeans.labels_
print(n_clusters)
# predict labels for test embeddings using k-means
test_labels = kmeans.predict(test_embeddings)

# calculate accuracy
accuracy = accuracy_score(test_ids, test_labels)
print(f"\n\nAccuracy: {accuracy:.2f}")