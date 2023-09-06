import os
import numpy as np

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


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def calculate_accuracy(test_embeddings, train_embeddings, threshold):
    correct_matches = 0
    total_tests = len(test_embeddings)

    for i, test_embedding in enumerate(test_embeddings):
        similarity_scores = []
        normalized_test = normalize_vector(test_embedding)
        
        for j, train_embedding in enumerate(train_embeddings):
            normalized_train = normalize_vector(train_embedding)
            dot_product = np.dot(normalized_test, normalized_train)
            similarity_scores.append(dot_product)

        max_similarity = np.max(similarity_scores)
        if max_similarity >= threshold:
            train_image_index = np.argmax(similarity_scores)
            print(f"Dot product similarity for test image {i+1}:")
            print(f"Best match is with train image {train_image_index+1}, with dot product similarity of {max_similarity:.4f}")
            print()

            correct_matches += 1

    accuracy = (correct_matches / total_tests) * 100
    return accuracy


# Load embeddings from train and test folders
train_embeddings = load_embeddings(train_folder_path)
test_embeddings = load_embeddings(test_folder_path)

# Set the threshold for dot product similarity
threshold = 0.96

# Calculate accuracy and print dot product similarity details
accuracy = calculate_accuracy(test_embeddings, train_embeddings, threshold)

print(f"Accuracy: {accuracy:.2f}%")
