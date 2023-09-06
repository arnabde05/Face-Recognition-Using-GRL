import os
import numpy as np

train_path = r"F:\Face_recognition\graph neural network\code\deepwalk\deepwalk\emb\train\90"
test_path = r"F:\Face_recognition\graph neural network\code\deepwalk\deepwalk\emb\test\75"

train_embeddings = []
train_ids = []
for filename in os.listdir(train_path):
    if filename.endswith(".embeddings"):
        emb_path = os.path.join(train_path, filename)
        train_embeddings.append(np.loadtxt(emb_path, usecols=[0]))
        train_id = filename.split("_")[0]
        train_ids.append(train_id)
train_embeddings = np.array(train_embeddings)


test_embeddings = []
test_ids = []
for filename in os.listdir(test_path):
    if filename.endswith(".embeddings"):
        emb_path = os.path.join(test_path, filename)
        test_embeddings.append(np.loadtxt(emb_path, usecols=[0]))
        test_id = filename.split("_")[0]
        test_ids.append(test_id)
test_embeddings = np.array(test_embeddings)

# Set the threshold for Euclidean distance similarity
threshold = 0.5

# Calculate Euclidean distance similarity for each test image with all train images
num_correct_matches = 0

for i, test_emb in enumerate(test_embeddings):
    print(f"\n\nEuclidean distance similarity for test image {i+1} (subject {test_ids[i]}):")
    distances = np.linalg.norm(test_emb - train_embeddings, axis=1)
    min_distance = np.min(distances)
    match_index = np.argmin(distances)
    match_id = train_ids[match_index]
    print(f"Best match is with train image {match_index+1} (subject {match_id}), with Euclidean distance similarity of {min_distance:.4f}")
    
    if min_distance <= threshold:
        if test_ids[i] == match_id:
            num_correct_matches += 1

accuracy = num_correct_matches / len(test_embeddings)
print(f"\n\nAccuracy: {accuracy:.2f}")
