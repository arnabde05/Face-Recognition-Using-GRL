import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Set paths for train and test embeddings
train_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\training\\150"
test_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\flattened_all"

# Read the embeddings for train and test images and store their respective IDs
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

# Define a range of values for k
k_range = range(1, 31)

# Cross-validate to find the best value of k
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, train_embeddings, train_ids, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find the optimal value of k
optimal_k = k_range[np.argmax(cv_scores)]
print(f"The optimal value of k is {optimal_k}")

# Train the KNN model with the optimal value of k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(train_embeddings, train_ids)

# Evaluate the model on the test set
accuracy = knn.score(test_embeddings, test_ids)
print(f"\n\nAccuracy: {accuracy:.2f}")
