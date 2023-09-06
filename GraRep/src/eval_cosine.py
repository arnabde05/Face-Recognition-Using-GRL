import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

train_path = r"F:\Face_recognition\graph neural network\code\GraRep\GraRep\Evaluation\train\150"
test_path = r"F:\Face_recognition\graph neural network\code\GraRep\GraRep\Evaluation\test\15"

# Read the embeddings for train and test images and store their respective IDs
train_embeddings = []
train_ids = []
for filename in os.listdir(train_path):
    if filename.endswith(".csv"):
        emb_path = os.path.join(train_path, filename)
        train_embeddings.append(pd.read_csv(emb_path, skiprows=1))
        train_id = filename.split("_")[0]
        train_ids.append(train_id)
train_embeddings = np.array(train_embeddings)
train_embeddings = np.reshape(train_embeddings, (train_embeddings.shape[0], -1))

test_embeddings = []
test_ids = []
for filename in os.listdir(test_path):
    if filename.endswith(".csv"):
        emb_path = os.path.join(test_path, filename)
        test_embeddings.append(pd.read_csv(emb_path, skiprows=1))
        test_id = filename.split("_")[0]
        test_ids.append(test_id)
test_embeddings = np.array(test_embeddings)
test_embeddings = np.reshape(test_embeddings, (test_embeddings.shape[0], -1))

# Set the threshold for cosine similarity
threshold = 0.8

# Calculate cosine similarity for each test image with all train images
num_correct_matches = 0

for i, test_emb in enumerate(test_embeddings):
    print(f"\n\nCosine similarity for test image {i+1} (subject {test_ids[i]}):")
    similarities = cosine_similarity([test_emb], train_embeddings)[0]
    max_sim = max(similarities)
    match_index = np.argmax(similarities)
    match_id = train_ids[match_index]
    print(f"Best match is with train image {match_index+1} (subject {match_id}), with cosine similarity of {max_sim:.4f}")
    
    if max_sim >= threshold and test_ids[i] == match_id:
        num_correct_matches += 1

accuracy = num_correct_matches / len(test_embeddings)
print(f"\n\nAccuracy: {accuracy:.2f}")
