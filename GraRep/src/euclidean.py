import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

train_folder_path = r"F:\Face_recognition\graph neural network\code\GraRep\GraRep\Evaluation\train\150"
test_folder_path = r"F:\Face_recognition\graph neural network\code\GraRep\GraRep\Evaluation\test\15"

def load_embeddings(folder_path):
    embeddings = []
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                embedding = np.loadtxt(file, delimiter=",", skiprows=1)  # Skip the header row
                embeddings.append(embedding)
    return embeddings

def calculate_accuracy(test_embeddings, train_embeddings, threshold):
    correct_matches = 0
    total_tests = len(test_embeddings)

    for i, test_embedding in enumerate(test_embeddings):
        train_embeddings_reshaped = np.array(train_embeddings).reshape(len(train_embeddings), -1)
        distance_scores = euclidean_distances(test_embedding.reshape(1, -1), train_embeddings_reshaped)
        min_distance = np.min(distance_scores)
        if min_distance <= threshold:
            train_image_index = np.argmin(distance_scores)
            print(f"Euclidean distance for test image {i+1} (subject {i+1}):")
            print(f"Best match is with train image {train_image_index+1} (subject {train_image_index+1}), with Euclidean distance of {min_distance:.4f}")
            print()

            correct_matches += 1

    accuracy = (correct_matches / total_tests) * 100
    return accuracy

# Load embeddings from train and test folders
train_embeddings = load_embeddings(train_folder_path)
test_embeddings = load_embeddings(test_folder_path)

# Set the threshold for Euclidean distance
threshold = 2.0

# Calculate accuracy and print Euclidean distance details
accuracy = calculate_accuracy(test_embeddings, train_embeddings, threshold)

print(f"Accuracy: {accuracy:.2f}%")
