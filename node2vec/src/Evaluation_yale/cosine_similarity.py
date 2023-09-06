import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# set the paths for train and test embeddings
train_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\training\\90"
test_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\testing\\75"

# read the embeddings for train and test images
train_embeddings = []
for filename in os.listdir(train_path):
    if filename.endswith(".emb"):
        emb_path = os.path.join(train_path, filename)
        train_embeddings.append(np.loadtxt(emb_path))
train_embeddings = np.array(train_embeddings)

test_embeddings = []
for filename in os.listdir(test_path):
    if filename.endswith(".emb"):
        emb_path = os.path.join(test_path, filename)
        test_embeddings.append(np.loadtxt(emb_path))
test_embeddings = np.array(test_embeddings)

# calculate cosine similarity for each test image with all train images
for i, test_emb in enumerate(test_embeddings):
    print(f"\n\nCosine similarity for test image {i+1}:")
    similarities = cosine_similarity([test_emb], train_embeddings)[0]
    for j, sim in enumerate(similarities):
        print(f"Train image {j+1}: {sim:.4f}")

similarity_matrix = cosine_similarity(test_embeddings, train_embeddings)
# print the similarity matrix
print(similarity_matrix)
