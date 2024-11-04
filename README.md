# PRODIGY_ML_03

# Image Classification of Cats and Dogs Using Support Vector Classifier

This project focuses on classifying images of cats and dogs using a Support Vector Classifier (SVC). The process involves several key steps, including data loading, preprocessing, model training, and evaluation.
## Project Overview

The dataset consists of images stored in two categories: Cats and Dogs. The images are loaded from a specified directory and processed to ensure consistency in size and format. Each image is resized to 50x50 pixels and converted to grayscale to simplify the classification task.
## Data Loading and Preprocessing

The project begins by importing necessary libraries such as OpenCV, NumPy, and Scikit-learn. The images from the directory are read and resized, and each image is flattened into a 1D array for easier manipulation. This data is then stored in a list along with the corresponding labels. If any errors occur during the loading process, they are ignored to maintain data integrity.
## Model Training

To enhance the performance of the classification model, the data is shuffled to ensure a random distribution of samples. The dataset is split into features and labels, followed by a train-test split where 85% of the data is used for training and 15% for testing. A Support Vector Classifier is initialized with a polynomial kernel and trained using the training dataset. Once the model is trained, it is saved for future use.
## Model Evaluation

The trained model is then loaded, and predictions are made on the test set. The model's accuracy is calculated, and a confusion matrix is generated to visualize the performance of the classification. This matrix illustrates the number of correct and incorrect predictions across the two categories (Cat and Dog).
## Visualization of Results

A classification report is generated to provide detailed metrics such as precision, recall, and F1-score for each category. Additionally, sample predictions are visualized by displaying five test images along with their predicted labels. This allows for a qualitative assessment of the model's performance.
## Conclusion

Through this project, the application of machine learning techniques for image classification is demonstrated, showcasing how SVC can effectively distinguish between cats and dogs. The methods employed ensure a robust approach to image classification, providing valuable insights into the model's capabilities and performance on unseen data.

## Dataset

The dataset used for this project can be found at the following link:https://www.kaggle.com/c/dogs-vs-cats/data
