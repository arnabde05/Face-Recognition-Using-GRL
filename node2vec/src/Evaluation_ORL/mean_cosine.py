import os
import numpy as np
from scipy.spatial.distance import cosine

train_path = "F:/Face_recognition/graph neural network/code/node2vec/node2vec/ORL_ALL/training_orl/369/"
test_path = "F:/Face_recognition/graph neural network/code/node2vec/node2vec/ORL_ALL/testing_orl/41/"
accuracy = 0 


for folder_name in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder_name)
    embeddings = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        embedding = np.loadtxt(file_path)
        embeddings.append(embedding)
    mean_embedding = np.mean(embeddings, axis=0)
    print("Mean embedding for folder", folder_name, "is", mean_embedding)

'''
for folder_name in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder_name)
    embeddings = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        embedding = np.loadtxt(file_path)
        embeddings.append(embedding)
    mean_embedding = np.mean(embeddings, axis=0)
    print("Mean embedding for folder", folder_name, "is", mean_embedding)
'''

for folder_name in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder_name)
    if not os.path.isdir(folder_path):
        continue
    embeddings = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue
        embedding = np.loadtxt(file_path)
        embeddings.append(embedding)
    mean_embedding = np.mean(embeddings, axis=0)
    print("Mean embedding for folder", folder_name, "is", mean_embedding)
