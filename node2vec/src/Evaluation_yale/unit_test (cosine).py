import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# set the paths for train and test embeddings
train_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\training\\150"
test_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\testing\\15"

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

# define the threshold for cosine distance
threshold = 0.4

# calculate cosine similarity for each test image with all train images
correct_matches = 0
for i, test_emb in enumerate(test_embeddings):
    similarities = cosine_similarity([test_emb], train_embeddings)[0]
    best_match_idx = np.argmin(similarities)
    if similarities[best_match_idx] < threshold:
        print(f"Test image {i+1} matches with train image {best_match_idx+1}")
        correct_matches += 1
    else:
        print(f"Test image {i+1} does not match with any train image")

# calculate accuracy
accuracy = correct_matches / len(test_embeddings)
print(f"\nAccuracy: {accuracy:.2f}")
