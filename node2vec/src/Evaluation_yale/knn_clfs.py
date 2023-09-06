import os 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load flattened embeddings
train_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\training\\120s"
test_path = "F:\\Face_recognition\\graph neural network\\code\\node2vec\\node2vec\\testing\\45s"

# read the embeddings for train and test images and store their respective IDs
train_embeddings = []
train_ids = []
for filename in os.listdir(train_path):
    if filename.endswith(".emb"):
        emb_path = os.path.join(train_path, filename)
        train_embeddings.append(np.loadtxt(emb_path))
        train_id = filename.split("_")[0]
        train_ids.append(train_id)
train_embeddings = np.array(train_embeddings)

test_embeddings = []
test_ids = []
for filename in os.listdir(test_path):
    if filename.endswith(".emb"):
        emb_path = os.path.join(test_path, filename)
        test_embeddings.append(np.loadtxt(emb_path))
        test_id = filename.split("_")[0]
        test_ids.append(test_id)
test_embeddings = np.array(test_embeddings)
# X = train_embeddings
# train_embeddings = np.load(train_path)

# Set the true labels for the flattened embeddings

true_labels = {"1_07_surprised_flattened.emb":"7",
               "2_15_centerlight_flattened.emb": "15",
               "3_12_surprised_flattened.emb":"12",
               "4_11_happy_flattened.emb":"11",
               "5_03_sleepy_flattened.emb":"3",
               "6_06_noglasses_flattened.emb":"6",
               "7_07_sleepy_flattened.emb":"7",
               "8_10_leftlight_flattened.emb":"10",
               "9_12_happy_flattened.emb":"12",
               "10_12_centerlight_flattened.emb":"12",
               "11_14_noglasses_flattened.emb":"14",
               "12_08_rightlight_flattened.emb":"8",
               "13_09_centerlight_flattened.emb": "9",
               "14_01_surprised_flattened.emb":"1",
               "15_05_leftlight_flattened.emb":"5",
               "16_08_sad_flattened.emb":"8",
               "17_13_sleepy_flattened.emb":"13",
               "18_01_sleepy_flattened.emb":"1",
               "19_07_wink_flattened.emb":"7",
               "20_08_happy_flattened.emb":"8",
               "21_07_glasses_flattened.emb":"7",
               "22_02_sad_flattened.emb":"2",
               "23_08_noglasses_flattened.emb":"8",
               "24_03_glasses_flattened.emb":"3",
               "25_01_centerlight_flattened.emb":"1",
               "26_04_glasses_flattened.emb":"4",
               "27_02_happy_flattened.emb":"2",
               "28_09_sleepy_flattened.emb":"9",
               "29_15_sleepy_flattened.emb":"15",
               "30_13_surprised_flattened.emb":"13",
               "31_06_leftlight_flattened.emb":"6",
               "32_surprised_06_sad_flattened.emb":"6",
               "33_11_sleepy_flattened.emb":"11",
               "34_05_glasses_flattened.emb":"5",
               "35_10_rightlight_flattened.emb":"10",
               "36_10_happy_flattened.emb":"10",
               "37_11_noglasses_flattened.emb":"11",
               "38_06_rightlight_flattened.emb":"6",
               "39_07_leftlight_flattened.emb":"7",
               "40_08_surprised_flattened.emb":"8",
               "41_12_leftlight_flattened.emb":"12",
               "42_09_leftlight_flattened.emb":"9",
               "43_11_leftlight_flattened.emb":"11",
               "44_03_normal_flattened.emb":"3",
               "45_11_wink_flattened.emb":"11",
               "46_12_sleepy_flattened.emb":"12",
               "47_01_sad_flattened.emb":"1",
               "48_05_sleepy_flattened.emb":"5",
               "49_02_normal_flattened.emb":"2",
               "50_09_normal_flattened.emb":"9",
               "51_15_leftlight_flattened.emb":"15",
               "52_07_noglasses_flattened.emb":"7",
               "53_11_surprised_flattened.emb":"11",
               "54_08_sleepy_flattened.emb":"8",
               "55_05_sad_flattened.emb":"5",
               "56_10_sad_flattened.emb":"10",
               "57_05_noglasses_flattened.emb":"5",
               "58_07_sad_flattened.emb":"7",
               "59_09_wink_flattened.emb":"9",
               "60_03_noglasses_flattened.emb":"3",
               "61_08_wink_flattened.emb":"8",
               "62_12_normal_flattened.emb":"12",
               "63_11_glasses_flattened.emb":"11",
               "64_03_sad_flattened.emb":"03",
               "65_13_leftlight_flattened.emb":"13",
               "66_15_happy_flattened.emb":"15",
               "67_14_glasses_flattened.emb":"14",
               "68_14_sad_flattened.emb":"14",
               "69_11_centerlight_flattened.emb":"11",
               "70_06_glasses_flattened.emb":"6",
               "71_06_sleepy_flattened.emb":"6",
               "72_14_sleepy_flattened.emb":"14",
               "73_14_normal_flattened.emb":"14",
               "74_15_sad_flattened.emb":"15",
               "75_04_sleepy_flattened.emb":"4",
               "76_07_centerlight_flattened.emb":"7",
               "77_01_wink_flattened.emb":"1",
               "78_04_leftlight_flattened.emb":"4",
               "79_04_normal_flattened.emb":"4",
               "80_14_surprised_flattened.emb":"14",
               "81_04_sad_flattened.emb":"4",
               "82_01_rightlight_flattened.emb":"1",
               "83_04_surprised_flattened.emb":"4",
               "84_03_happy_flattened.emb":"3",
               "85_09_rightlight_flattened.emb":"9",
               "86_06_normal_flattened.emb":"6",
               "87_14_wink_flattened.emb":"14",
               "88_02_leftlight_flattened.emb":"2",
               "89_15_rightlight_flattened.emb":"15",
               "90_15_noglasses_flattened.emb":"15",
               "91_12_glasses_flattened.emb":"12",
               "92_14_centerlight_flattened.emb":"14",
               "93_14_rightlight_flattened.emb":"14",
               "94_13_normal_flattened.emb":"13",
               "95_01_noglasses_flattened.emb":"1",
               "96_08_leftlight_flattened.emb":"8",
               "97_07_rightlight_flattened.emb":"7",
               "98_13_wink_flattened.emb":"13",
               "99_01_leftlight_flattened.emb":"1",
               "100_09_surprised_flattened.emb":"9",
               "101_03_wink_flattened.emb":"3",
               "102_13_rightlight_flattened.emb":"13",
               "103_02_surprised_flattened.emb":"2",
               "104_13_glasses_flattened.emb":"13",
               "105_10_wink_flattened.emb":"10",
               "106_07_normal_flattened.emb":"7",
               "107_12_sad_flattened.emb":"12",
               "108_05_centerlight_flattened.emb":"5",
               "109_01_happy_flattened.emb":"1",
               "110_15_wink_flattened.emb":"15",
               "111_05_surprised_flattened.emb":"5",
               "112_08_normal_flattened.emb":"8",
               "113_09_happy_flattened.emb":"9",
               "114_10_sleepy_flattened.emb":"10",
               "115_05_normal_flattened.emb":"5",
               "116_13_sad_flattened.emb":"13",
               "117_06_centerlight_flattened.emb":"6",
               "118_10_glasses_flattened.emb":"10",
               "119_10_centerlight_flattened.emb":"10",
               "120_05_wink_flattened.emb":"5",}


test_labels = { 
              "121_13_happy_flattened.emb":"13",
               "122_04_rightlight_flattened.emb":"4",
               "123_10_noglasses_flattened.emb":"10",
               "124_04_wink_flattened.emb":"4",
               "125_05_rightlight_flattened.emb":"5",
               "126_05_happy_flattened.emb":"5",
               "127_11_normal_flattened.emb":"11",
               "128_12_wink_flattened.emb":"12",
               "129_12_rightlight_flattened.emb":"12",
               "130_11_rightlight_flattened.emb":"11",
               "131_08_centerlight_flattened.emb":"8",
               "132_04_centerlight_flattened.emb":"4",
               "133_03_rightlight_flattened.emb":"3",
               "134_01_glasses_flattened.emb":"1",
               "135_02_rightlight_flattened.emb":"2",
               "136_14_happy_flattened.emb":"14",
               "137_15_glasses_flattened.emb":"15",
               "138_04_happy_flattened.emb":"4",
               "139_02_noglasses_flattened.emb":"2",
               "140_15_normal_flattened.emb":"15",
               "141_06_wink_flattened.emb":"6",
               "142_09_noglasses_flattened.emb":"9",
               "143_11_sad_flattened.emb":"11",
               "144_01_normal_flattened.emb":"1",
               "145_02_centerlight_flattened.emb":"2",
               "146_10_normal_flattened.emb":"10",
               "147_03_surprised_flattened.emb":"3",
               "148_02_glasses_flattened.emb":"2",
               "149_15_surprised_flattened.emb":"15",
               "150_09_sad_flattened.emb":"9",
               "151_06_surprised_flattened.emb":"6",
               "152_13_centerlight_flattened.emb":"13",
               "153_14_leftlight_flattened.emb":"14",
               "154_02_wink_flattened.emb":"2",
               "155_10_surprised_flattened.emb":"10",
               "156_03_centerlight_flattened.emb":"3",
               "157_04_noglasses_flattened.emb":"4",
               "158_02_sleepy_flattened.emb":"2",
               "159_09_glasses_flattened.emb":"9",
               "160_08_glasses_flattened.emb":"8",
               "161_07_happy_flattened.emb":"7",
               "162_06_happy_flattened.emb":"6",
               "163_03_leftlight_flattened.emb":"3",
               "164_12_noglasses_flattened.emb":"12",
               "165_13_noglasses_flattened.emb":"13"}
              


training_labels = np.array(list(true_labels.values()), dtype=int)
test_labels = np.array(list(test_labels.values()), dtype=int)

print(len(training_labels))

print(len(test_labels))
# y = true_labels
# Create a kNN classifier with k=2
k = 1
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the kNN classifier with the flattened embeddings and true labels
knn_classifier.fit(train_embeddings, training_labels)

# Use the kNN classifier to predict the labels of the flattened embeddings
predicted_labels = knn_classifier.predict(test_embeddings)

# print(predicted_labels)
# print (test_labels)
print(training_labels)
# Print the accuracy of the kNN classifier
accuracy = accuracy_score(test_labels, predicted_labels)
print ("Accuracy : ", accuracy)


cm = confusion_matrix(test_labels, predicted_labels)

sns.heatmap(cm)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# show plot
plt.show()