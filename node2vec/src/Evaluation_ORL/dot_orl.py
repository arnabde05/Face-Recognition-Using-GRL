import os
import numpy as np

# set the paths for train and test embeddings
train_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\ORL_ALL\\training_orl\\63"
test_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\ORL_ALL\\testing_orl\\42"

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

# calculate dot product for each test image with all train images
num_correct_matches = 0
for i, test_emb in enumerate(test_embeddings):
    print(f"\n\nDot product for test image {i+1} (subject {test_ids[i]}):")
    dot_products = np.dot(test_emb, train_embeddings.T)
    max_dot = max(dot_products)
    match_index = np.argmax(dot_products)
    match_id = train_ids[match_index]
    print(f"Best match is with train image {match_index+1} (subject {match_id}), with dot product of {max_dot:.4f}")
    if test_ids[i] == match_id:
        num_correct_matches += 1

accuracy = num_correct_matches / len(test_embeddings)
print(f"\n\nAccuracy: {accuracy:.2f}")