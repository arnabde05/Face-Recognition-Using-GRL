import numpy as np
import networkx as nx
from node2vec import Node2Vec

# Load adjacency matrix from txt file
adj_mat = np.loadtxt("F:/Face_recognition/graph neural network/code/node2vec/node2vec/adjacency_matrix.txt")

# Convert to NetworkX graph
G = nx.from_numpy_array(adj_mat)

# Generate node embeddings with node2vec
node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200)
model = node2vec.fit(window=10, min_count=1)
embeddings = model.wv.vectors

# Save model
model.wv.save_word2vec_format("F:/Face_recognition/graph neural network/code/node2vec/node2vec/emb_output/subject01_normal.emb")