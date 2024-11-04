# Import necessary libraries
import os
import cv2  # OpenCV library for image processing
import numpy as np
import pickle  # For saving/loading data
import random  # For shuffling data
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.svm import SVC  # Support Vector Classifier for classification
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For visualization (confusion matrix)
from sklearn.metrics import confusion_matrix, classification_report  # For model evaluation

# Load and preprocess data
dir = 'D:\\Pycharm Projects\\prodgy_ml_task_03\\PetImages'
categories = ['Cat', 'Dog']
data = []

# Loop through each category (Cat and Dog)
for category in categories:
    path = os.path.join(dir, category)

    # Load images from each category's directory
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath, 0)  # Read image in grayscale
        try:
            pet_img = cv2.resize(pet_img, (50, 50))  # Resize image to 50x50 for consistency
            image = np.array(pet_img).flatten()  # Flatten the image to a 1D array

            data.append([image, label])  # Append flattened image and its label to data list
        except Exception as e:
            pass  # Ignore any errors during loading

# Check data length
print(len(data))

# Load preprocessed data from a .pickle file
pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)  # Load data
pick_in.close()

# Print the number of samples in data to verify loading
print(f"Number of samples in data: {len(data)}")

# Shuffle data to ensure random distribution of samples
random.shuffle(data)
features = []
labels = []

# Split data into features and labels
for feature, label in data:
    features.append(feature)
    labels.append(label)

# Split data into training and testing sets (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15)

# Uncomment these lines to train a new model
# model = SVC(C=1, kernel='poly', gamma='auto')  # Initialize SVM model with polynomial kernel
# model.fit(X_train, y_train)  # Train the model with training data
#
# # Save trained model using pickle
# pick = open('model.sav', 'wb')
# pickle.dump(model, pick)
# pick.close()
# print("Model is saved")

# Load previously saved model from .sav file
pick = open('model.sav', 'rb')
model = pickle.load(pick)  # Load trained model
pick.close()

# Calculate predictions and accuracy on the test set
prediction = model.predict(X_test)
accuracy = model.score(X_test, y_test)  # Model accuracy on test set
categories = ['Cat', 'Dog']
print("Accuracy: ", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, prediction)  # Compute confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, prediction, target_names=categories))

# Visualize Sample Predictions
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.imshow(np.array(X_test[i]).reshape(50, 50), cmap='gray')  # Reshape and display test images
    ax.set_title(f"Predicted: {categories[prediction[i]]}")  # Display predicted label
    ax.axis('off')
plt.suptitle("Sample Predictions")
plt.show()
