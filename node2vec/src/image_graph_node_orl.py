import os

import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import mediapipe as mp

# Create FaceMesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Set parameters for node2vec
dimensions = 16
walk_length = 30
num_walks = 200
window_size = 10
min_count = 1

# Set threshold for adjacency matrix
threshold = 0.1  # You may need to adjust this threshold

# Set input and output directories
input_dir = "F:/Face_recognition/graph neural network/code/node2vec/node2vec/ORL_ALL/orl(20)"
output_dir = "F:/Face_recognition/graph neural network/code/node2vec/node2vec/ORL_ALL/orl20_output"

# Loop over all images in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".jpg"):
        # Read image
        image_path = os.path.join(input_dir, file_name)
        image = cv2.imread(image_path)

        # Process image
        result = face_mesh.process(image)

        # Get facial landmarks
        if result.multi_face_landmarks:
            facial_landmarks = result.multi_face_landmarks[0]

            # Compute the Euclidean distances between all pairs of landmark points
            n_landmarks = len(facial_landmarks.landmark)
            landmarks = np.zeros((n_landmarks, 2))
            for i, landmark in enumerate(facial_landmarks.landmark):
                landmarks[i] = [landmark.x, landmark.y]
            distances = np.linalg.norm(landmarks[:, np.newaxis, :] - landmarks[np.newaxis, :, :], axis=-1)

            # Convert distances to adjacency matrix
            adj_mat = (distances <= threshold).astype(int)

            # Calculate number of nodes and edges
            n_nodes = adj_mat.shape[0]
            n_edges = np.count_nonzero(adj_mat) // 2

            # Convert to NetworkX graph
            G = nx.from_numpy_array(adj_mat)

            # Generate node embeddings with node2vec
            node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)
            model = node2vec.fit(window=window_size, min_count=min_count)
            embeddings = model.wv.vectors

            # Flatten embeddings
            flattened_embeddings = embeddings.flatten()

            # Save flattened embeddings to file with the same subject name in the input directory
            output_file_name = file_name.replace(".jpg", ".emb")
            output_path = os.path.join(output_dir, output_file_name)
            np.savetxt(output_path, flattened_embeddings)

            print(f"Processed image {file_name}, saved embeddings to {output_path}")
        else:
            print(f"No face detected in image {file_name}")
