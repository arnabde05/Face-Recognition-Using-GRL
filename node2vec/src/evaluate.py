import os
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the directory containing the face image embeddings
face_emb_dir = "F:/Face_recognition/graph neural network/code/node2vec/node2vec/flattened_emb_output"

# Path to the test image embedding file
test_emb_file = "F:/Face_recognition/graph neural network/code/node2vec/node2vec/test_emb/subject08_sad_flattened_output.emb"

# Define the classes and their corresponding filenames
class_filenames = {
    "class01": ["01_sleepy_flattened_output.emb"],
    "class02": ["02_centerlight_flattened_output.emb"],
    "class03": ["03_glasses_flattened_output.emb"],
    "class04": ["04_sad_flattened_output.emb"],
    "class05": ["05_surprised_flattened_output.emb"],
    "class06": ["06_centerlight_flattened_output.emb"],
    "class07": ["07_happy_flattened_output.emb"],
    "class08": ["08_leftlight_flattened_output.emb"],
    "class09": ["09_rightlight_flattened_output.emb"],
    "class10": ["10_noglasses_flattened_output.emb"],
    "class11": ["11_leftlight_flattened_output.emb"],
    "class12": ["12_happy_flattened_output.emb"],
    "class13": ["13_normal_flattened_output.emb"],
    "class14": ["14_sleepy_flattened_output.emb"],
    "class15": ["15_centerlight_flattened_output.emb"]
}

# Load the test image embedding as a numpy array
test_emb = np.loadtxt(test_emb_file)

# Calculate the cosine similarity between the test image embedding and each class's set of embeddings
cosine_similarities = {}
for class_name, filenames in class_filenames.items():
    class_embs = []
    for filename in filenames:
        emb_file = os.path.join(face_emb_dir, filename)
        emb = np.loadtxt(emb_file)
        class_embs.append(emb)
    class_similarities = [1 - cosine(test_emb, emb) for emb in class_embs]  # cosine similarity is 1 - cosine distance
    cosine_similarities[class_name] = np.mean(class_similarities)

# Print the resulting cosine similarities for each class
for class_name, similarity in cosine_similarities.items():
    print(class_name, similarity)

# Choose the class with the highest average cosine similarity as the predicted class
predicted_class = max(cosine_similarities, key=cosine_similarities.get)
print("Predicted class:", predicted_class)

# Define the actual classes
actual_classes = ["class01", "class02", "class03", "class04", "class05", "class06", "class07", "class08", "class09", "class10", "class11", "class12", "class13", "class14", "class15"]

# Define the predicted classes
predicted_classes = [predicted_class]*len(actual_classes)

# Create the confusion matrix
conf_matrix = confusion_matrix(actual_classes, predicted_classes, labels=actual_classes)

sns.set(font_scale=0.7)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=predicted_classes, yticklabels=actual_classes)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Print the confusion matrix
print(conf_matrix)