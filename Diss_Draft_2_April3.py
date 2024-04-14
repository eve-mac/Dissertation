import os
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def read_tif_files(folder_path):
    tif_data = {}
    for document_type in os.listdir(folder_path):
        document_folder_path = os.path.join(folder_path, document_type)
        tif_files = [os.path.join(document_folder_path, file) for file in os.listdir(document_folder_path) if file.endswith('.tif')]
        tif_data[document_type] = tif_files
    return tif_data

def combine_data(tif_data):
    combined_data = []
    for document_type, files in tif_data.items():
        for file in files:
            combined_data.append((file, document_type))
    return combined_data

def split_data(combined_data, test_size=0.2):
    random.shuffle(combined_data)
    X = [data[0] for data in combined_data]
    y = [data[1] for data in combined_data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    folder_path = 'Datasets/online_data'  # Assuming 'online_data' is the base folder containing subfolders for each document type
    tif_data = read_tif_files(folder_path)
    combined_data = combine_data(tif_data)
    X_train, X_test, y_train, y_test = split_data(combined_data)
    
    # Train Naive Bayes classifier
    nb_classifier = MultinomialNB()
    X_train_counts = [[1] for _ in range(len(X_train))]  # Placeholder feature vectors
    nb_classifier.fit(X_train_counts, y_train)
    
    # Test classifier
    X_test_counts = [[1] for _ in range(len(X_test))]  # Placeholder feature vectors
    y_pred = nb_classifier.predict(X_test_counts)
    accuracy = accuracy_score(y_test, y_pred)
    print("Naive Bayes Classifier Accuracy:", accuracy)

if __name__ == "__main__":
    main()
