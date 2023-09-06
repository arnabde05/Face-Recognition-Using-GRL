import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# set the paths for train and test embeddings
train_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\training\\120s"
test_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\testing\\45s"

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

# define the KNN model
k = 1
knn = KNeighborsClassifier(n_neighbors=k)

# fit the KNN model on the train embeddings and labels
knn.fit(train_embeddings, train_ids)

# predict the labels for the test embeddings
pred_ids = knn.predict(test_embeddings)

# calculate the accuracy
accuracy = accuracy_score(test_ids, pred_ids)
print(f"Accuracy using KNN: {accuracy:.2f}")
