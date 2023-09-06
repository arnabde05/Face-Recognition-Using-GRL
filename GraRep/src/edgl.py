import os
import cv2
import mediapipe as mp
import numpy as np
import networkx as nx
import csv

mp_drawing = mp.solutions.drawing_utils

# Set directories
img_dir = "F:/Face_recognition/graph neural network/code/GraRep/GraRep/yale_jpg"
output_dir = "F:/Face_recognition/graph neural network/code/GraRep/GraRep/edgelist1"

# Set conditions and image size
conditions = ["centerlight", "glasses", "happy", "sad", "leftlight", "sleepy", "normal", "noglasses", "rightlight", "surprised", "wink"]
img_size = (160, 160)

# Create FaceMesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Iterate over subject folders
for subject_folder in os.listdir(img_dir):
    if not os.path.isdir(os.path.join(img_dir, subject_folder)):
        continue
    
    # Get subject ID
    subject_id = subject_folder.replace("subject", "")
    
    # Iterate over conditions
    for condition in conditions:
        # Load image
        img_path = os.path.join(img_dir, subject_folder, f"{subject_folder}_{condition}.jpg")
        if not os.path.exists(img_path):
            continue
        image = cv2.imread(img_path)
        
        # Resize image
        image = cv2.resize(image, img_size)

        # Process image
        result = face_mesh.process(image)

        # Get facial landmarks
        facial_landmarks = result.multi_face_landmarks[0]

        # Compute the Euclidean distances between all pairs of landmark points
        n_landmarks = len(facial_landmarks.landmark)
        landmarks = np.zeros((n_landmarks, 2))
        for i, landmark in enumerate(facial_landmarks.landmark):
            landmarks[i] = [landmark.x, landmark.y]
        distances = np.linalg.norm(landmarks[:, np.newaxis, :] - landmarks[np.newaxis, :, :], axis=-1)

        # Convert distances to adjacency matrix
        threshold = 0.1  # You may need to adjust this threshold
        adj_mat = (distances <= threshold).astype(int)

        # Calculate number of nodes and edges
        n_nodes = adj_mat.shape[0]
        n_edges = np.count_nonzero(adj_mat) // 2

        # Convert to NetworkX graph
        G = nx.Graph()
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj_mat[i, j] == 1:
                    G.add_edge(i, j)

        # Write edge list to CSV file
        output_file = os.path.join(output_dir, f"{subject_folder}_{condition}.csv")
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id_1', 'id_2'])
            for edge in G.edges:
                writer.writerow(edge)
