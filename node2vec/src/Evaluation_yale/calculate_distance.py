from scipy.spatial.distance import cosine
import numpy as np

# Load the two embeddings
emb1 = np.loadtxt("F:/Face_recognition/graph neural network/code/node2vec/node2vec/flattened_emb_output/08_leftlight_flattened_output.emb")
emb2 = np.loadtxt("F:/Face_recognition/graph neural network/code/node2vec/node2vec/test_emb/subject08_sad_flattened_output.emb")

# Calculate the cosine distance between the two embeddings
cosine_distance = cosine(emb1, emb2)

# Print the resulting cosine distance
print("Cosine distance:", cosine_distance)