import os
import numpy as np
from scipy.spatial.distance import cosine

train_path = "F:/Face_recognition/graph neural network/code/node2vec/node2vec/ORL_ALL/training_orl/369/"
test_path = "F:/Face_recognition/graph neural network/code/node2vec/node2vec/ORL_ALL/testing_orl/41/"

# calculate mean embeddings for each folder in the training set
mean_embeddings = {}
for folder_name in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder_name)
    embeddings = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        embedding = np.loadtxt(file_path)
        embeddings.append(embedding)
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embeddings[folder_name] = mean_embedding

# calculate cosine distances between mean embeddings and test set embeddings
for folder_name in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder_name)
    if not os.path.isdir(folder_path):
        continue
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue
        embedding = np.loadtxt(file_path)
        min_distance = 2  # set initial minimum distance to be greater than 1 (i.e., greater than the maximum possible cosine distance)
        best_match = None
        for subject, mean_embedding in mean_embeddings.items():
            distance = cosine(mean_embedding, embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = subject
        print(f"Test image {file_name} best matches subject {best_match} with distance {min_distance:.2f}")
