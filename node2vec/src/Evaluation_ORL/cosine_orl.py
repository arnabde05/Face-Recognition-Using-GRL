import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# set the paths for train and test embeddings
train_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\ORL_ALL\\training_orl\\63"
test_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\ORL_ALL\\testing_orl\\42"

# Load the embeddings of the training set
train_embeddings = []
# train_filenames = []
for filename in os.listdir(train_path):
    filepath = os.path.join(train_path, filename)
    embedding = np.loadtxt(filepath)
    train_embeddings.append(embedding)
    # train_filenames.append(filename)

# print (embedding.shape)
# print (train_embeddings)

# Load the embeddings of the test set
test_embeddings = []
for filename in os.listdir(test_path):
    filepath = os.path.join(test_path, filename)
    embeddings = np.loadtxt(filepath)
    test_embeddings.append(embeddings)

# print(embeddings.shape)

'''
# Calculate the cosine distance between each test embedding and all training embeddings
for i, test_embedding in enumerate(test_embeddings):
    distances = cosine_similarity(test_embedding.reshape(1, -1), train_embeddings)
    print("Distances for test embedding {}: {}".format(i+1, distances))

'''

# Calculate the cosine distance between each test embedding and all training embeddings
max_similarities = []
max_similarity_indices = []
for i, test_embedding in enumerate(test_embeddings):
    distances = cosine_similarity(test_embedding.reshape(1, -1), train_embeddings)
    max_similarity_index = np.argmax(distances)
    max_similarity = np.max(distances)
    max_similarities.append(max_similarity)
    max_similarity_indices.append(max_similarity_index)

'''
# Print the maximum cosine similarity values for each test embedding
for i, max_similarity in enumerate(max_similarities):
    print("Max similarity for test embedding {}: {}".format(i+1, max_similarity))

'''

# Map the maximum cosine similarity values with the corresponding training embeddings
for i, max_similarity_index in enumerate(max_similarity_indices):
    max_similarity = max_similarities[i]
    max_similarity_train_embedding = train_embeddings[max_similarity_index]
    message = "Test embedding {} has max similarity of {} with training embedding {}"
    print(message.format(i+1, max_similarity, max_similarity_index+1))


# Map the maximum cosine similarity values with the corresponding training embeddings
threshold = 0.4
num_correct = 0
for i, max_similarity_index in enumerate(max_similarity_indices):
    max_similarity = max_similarities[i]
    if max_similarity > threshold:
        num_correct += 1
    max_similarity_train_embedding = train_embeddings[max_similarity_index]
    message = "Test embedding {} has max similarity of {} with training embedding {}"
    print(message.format(i+1, max_similarity, max_similarity_index+1))

# Calculate accuracy under the threshold
accuracy = num_correct / len(test_embeddings)
print("Accuracy under threshold {}: {:.2f}%".format(threshold, accuracy * 100))