import os
import numpy as np

train_path = r"F:\Face_recognition\graph neural network\code\GraRep\GraRep\Evaluation\train\150"
test_path = r"F:\Face_recognition\graph neural network\code\GraRep\GraRep\Evaluation\test\15"

# Read the embeddings for train and test images and store their respective IDs
train_embeddings = []
train_ids = []
for filename in os.listdir(train_path):
    if filename.endswith(".csv"):
        emb_path = os.path.join(train_path, filename)
        train_embeddings.append(np.loadtxt(emb_path, delimiter=",", skiprows=1))
        train_id = filename.split("_")[0]
        train_ids.append(train_id)
train_embeddings = np.array(train_embeddings)

test_embeddings = []
test_ids = []
for filename in os.listdir(test_path):
    if filename.endswith(".csv"):
        emb_path = os.path.join(test_path, filename)
        test_embeddings.append(np.loadtxt(emb_path, delimiter=",", skiprows=1))
        test_id = filename.split("_")[0]
        test_ids.append(test_id)
test_embeddings = np.array(test_embeddings)

# Reshape train and test embeddings for dot product calculation
train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], -1)

# Set the threshold for dot product similarity
threshold = 0.5

# Calculate dot product similarity for each test image with all train images
num_correct_matches = 0

for i, test_emb in enumerate(test_embeddings):
    print(f"\n\nDot product similarity for test image {i+1} (subject {test_ids[i]}):")
    similarities = np.dot(test_emb, train_embeddings.T)
    max_sim = np.max(similarities)
    match_indices = np.argmax(similarities)
    match_ids = train_ids[match_indices]
    print(f"Best match is with train image {match_indices+1} (subject {match_ids}), with dot product similarity of {max_sim:.4f}")
    
    if max_sim >= threshold and test_ids[i] == match_ids:
        num_correct_matches += 1

accuracy = num_correct_matches / len(test_embeddings)
print(f"\n\nAccuracy: {accuracy:.2f}")
