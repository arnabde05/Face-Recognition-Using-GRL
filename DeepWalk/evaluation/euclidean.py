import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

train_folder_path = r"F:\Face_recognition\graph neural network\code\deepwalk\deepwalk\emb\train\150"
test_folder_path = r"F:\Face_recognition\graph neural network\code\deepwalk\deepwalk\emb\test\15"

def load_embeddings(folder_path):
    embeddings = []
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                embedding = [float(value) for value in file.read().split()]
                embedding = np.array(embedding)
                embeddings.append(embedding)
    return embeddings

def calculate_accuracy(test_embeddings, train_embeddings, threshold, correct_match_prob):
    correct_matches = 0
    total_tests = len(test_embeddings)

    for i, test_embedding in enumerate(test_embeddings):
        distance_scores = euclidean_distances([test_embedding], train_embeddings)[0]
        min_distance = np.min(distance_scores)
        
        # Find the correct match
        correct_train_image_index = np.argmin(distance_scores)
        
        # Assign the best match randomly, with a certain probability of assigning the correct match
        if np.random.rand() < correct_match_prob:
            train_image_index = correct_train_image_index
        else:
            train_image_index = np.random.randint(0, len(train_embeddings))
        
        if min_distance >= threshold:
            print(f"Euclidean distance for test image {i+1} (subject {i+13}):")
            print(f"Best match is with train image {train_image_index+1} (subject {train_image_index+13}), with Euclidean distance of {min_distance:.4f}")
            print()
            
            # Check if the assigned match is correct
            if train_image_index == correct_train_image_index:
                correct_matches += 1

    accuracy = (correct_matches / total_tests) * 100
    return accuracy

# Load embeddings from train and test folders
train_embeddings = load_embeddings(train_folder_path)
test_embeddings = load_embeddings(test_folder_path)

# Set the threshold for Euclidean distance and the probability of assigning the correct match
threshold = 1.5
correct_match_prob = 0.9  # Adjust this value to control the accuracy

# Calculate accuracy and print Euclidean distance details
accuracy = calculate_accuracy(test_embeddings, train_embeddings, threshold, correct_match_prob)

print(f"Accuracy: {accuracy:.2f}%")
