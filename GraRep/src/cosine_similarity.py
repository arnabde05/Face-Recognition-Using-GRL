import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

train_folder_path = r"F:\Face_recognition\graph neural network\code\GraRep\GraRep\Evaluation\train\150"
test_folder_path = r"F:\Face_recognition\graph neural network\code\GraRep\GraRep\Evaluation\test\15"

def load_embeddings(folder_path):
    embeddings = []
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".csv"):
            embedding = np.loadtxt(file_path, delimiter=",", skiprows=1)  # Skip the header row
            embeddings.append(embedding)
    return embeddings


def calculate_accuracy(test_embeddings, train_embeddings, threshold):
    correct_matches = 0
    total_tests = len(test_embeddings)

    for i, test_embedding in enumerate(test_embeddings):
        similarity_scores = cosine_similarity(np.array(test_embedding).reshape(1, -1), np.array(train_embeddings).reshape(len(train_embeddings), -1)).flatten()
        max_similarity = np.max(similarity_scores)
        if max_similarity >= threshold:
            train_image_index = np.argmax(similarity_scores)
            print(f"Cosine similarity for test image {i+1}:")
            print(f"Best match is with train image {train_image_index+1} , with cosine similarity of {max_similarity:.4f}")
            print()

            correct_matches += 1

    accuracy = (correct_matches / total_tests) * 100
    return accuracy


# Load embeddings from train and test folders
train_embeddings = load_embeddings(train_folder_path)
test_embeddings = load_embeddings(test_folder_path)

# Set the threshold for cosine similarity
threshold = 0.953

# Calculate accuracy and print cosine similarity details
accuracy = calculate_accuracy(test_embeddings, train_embeddings, threshold)

print(f"Accuracy: {accuracy:.2f}%")