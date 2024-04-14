import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def read_image_files(folder_path):
    image_data = []
    labels = []
    for label, document_type in enumerate(os.listdir(folder_path)):
        document_folder_path = os.path.join(folder_path, document_type)
        for filename in os.listdir(document_folder_path):
            if filename.endswith('.tif'):
                file_path = os.path.join(document_folder_path, filename)
                # Load image data and preprocess (e.g., resize, normalize)
                image = Image.open(file_path)
                image = image.resize((224, 224))  # Resize image to a fixed size
                image = np.array(image) / 255.0  # Normalize pixel values
                image_data.append(image)
                labels.append(label)
    return np.array(image_data), np.array(labels)


def extract_text_from_tif(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text


def combine_data(tif_data):
    combined_data = []
    for document_type, files in tif_data.items():
        for file in files:
            document_text = extract_text_from_tif(file)
            combined_data.append((document_text, document_type))
    return combined_data


def preprocess_text(texts):
    # Convert text to lowercase
    texts = [text.lower() for text in texts]

    # Remove special characters and digits
    texts = [re.sub(r'[^a-zA-Z\s]', '', text) for text in texts]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    texts = [' '.join([word for word in text.split() if word not in stop_words]) for text in texts]

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Padding sequences to ensure uniform length
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_sequences, tokenizer.word_index


def combine_data_images_text(tif_data, image_data):
    combined_data = []
    idx = 0
    for document_type, files in tif_data.items():
        for file in files:
            combined_data.append((image_data[idx], document_type))
            idx += 1
    return combined_data


def split_data(combined_data, test_size=0.2):
    random.shuffle(combined_data)
    X = [data[0] for data in combined_data]
    y = [data[1] for data in combined_data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    folder_path = 'Datasets/online_data'  # Assuming 'online_data' is the base folder containing subfolders for each document type

    # Read image files and their labels
    image_data, labels = read_image_files(folder_path)

    # Read TIFF files and combine data
    combined_text_data = combine_data(read_tif_files(folder_path))

    # Combine image and text data
    combined_data = combine_data_images_text(read_tif_files(folder_path), image_data)

    # Data exploration
    print("Total number of images:", len(image_data))
    print("Number of classes:", len(np.unique(labels)))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(combined_data)

    # Build and train the model
    input_shape = X_train[0].shape
    num_classes = len(np.unique(y_train))
    model = build_model(input_shape, num_classes)
    model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test), verbose=2)
    print('\nTest accuracy:', test_acc)


if __name__ == "__main__":
    main()
