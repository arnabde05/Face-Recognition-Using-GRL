import cv2
import mediapipe as mp
import numpy as np
import networkx as nx
from node2vec import Node2Vec

mp_drawing = mp.solutions.drawing_utils

# Load image
image = cv2.imread("F:/Face_recognition/graph neural network/code/node2vec/node2vec/yale_jpg/subject01/subject01_centerlight.jpg")

# Create FaceMesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

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



# Save adjacency matrix to file
# np.savetxt("adjacency_matrix.txt", adj_mat, fmt="%d")


# Calculate number of nodes and edges
n_nodes = adj_mat.shape[0]
n_edges = np.count_nonzero(adj_mat) // 2

# Print results
print("Number of nodes:", n_nodes)
print("Number of edges:", n_edges)

# Display result
annotated_image = image.copy()
mp_drawing.draw_landmarks(annotated_image, facial_landmarks)
cv2.imshow("Result", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Convert to NetworkX graph

G = nx.from_numpy_array(adj_mat)

# Generate node embeddings with node2vec

node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200)
model = node2vec.fit(window=10, min_count=1)
embeddings = model.wv.vectors

print(embeddings.shape)

# Save model
# model.wv.save_word2vec_format("F:/Face_recognition/graph neural network/code/node2vec/node2vec/emb_output/subject01_normal.emb")

# Flatten embeddings
flattened_embeddings = embeddings.flatten()

# Save flattened embeddings to file
# np.savetxt("F:/Face_recognition/graph neural network/code/node2vec/node2vec/1_1.emb", flattened_embeddings)