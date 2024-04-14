import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pytesseract
from PIL import Image

#os and re seem to be inbuilt python - if errors check this
#sklearn depreciated, replaced with scikit-learn


# Step 1: Extract text from TIFF files
def extract_text_from_tif(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text

# Step 2: Create word clouds and other visualizations
def create_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Step 3: Preprocessing for ML algorithms
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)

    # Padding sequences to ensure uniform length
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_sequences, tokenizer.word_index

# Step 4: Create a Naive Bayes classifier
def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    predictions = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Step 5: Create an NLP deep learning algorithm
def create_deep_learning_model(input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_length))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to evaluate model accuracy
def evaluate_model(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy

# Function to generate histogram of word frequencies
def generate_word_frequency_histogram(text):
    word_freq = pd.Series(' '.join(text).split()).value_counts()[:30]
    plt.figure(figsize=(10,5))
    word_freq.plot(kind='bar', color='skyblue')
    plt.title('Top 30 Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()

# Function to generate bar chart of most common words
def generate_bar_chart_common_words(text):
    word_freq = pd.Series(' '.join(text).split()).value_counts()[:30]
    plt.figure(figsize=(10,5))
    plt.bar(word_freq.index, word_freq.values, color='lightcoral')
    plt.title('Bar Chart of Most Common Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

# Main body of the code
def main():

    folder = 'Datasets/online_data'  # Assuming 'online_data' is the base folder containing subfolders for each document type
    #document_types = [subfolder for subfolder in os.listdir(folder) if not subfolder.startswith('.')]
    document_types = ['scientific_report', 'specification']
    # Get the list of document types (subfolders)
    print(document_types)

    # Loop through each document type folder
    for document_type in document_types:
        document_folder_path = os.path.join(folder, document_type)  # Fixed variable name and folder path
        document_text = ""  # Initialize empty string to store text for the current document type
        
        # Loop through each file in the document type folder
        for filename in os.listdir(document_folder_path):
            file_path = os.path.join(document_folder_path, filename)
            text = extract_text_from_tif(file_path)
            document_text += text + " "  # Append text to the document_text
        
        # Print the document type
        print("Document Type:", document_type)

        # After processing all files in the document type folder, create a word cloud
        create_word_cloud(document_text, document_type)  # Added title parameter for word cloud title

    # Example usage of preprocessing text
    # Assume 'text' is your loaded text data
    #X, word_index = preprocess_text(text)
    #y = labels

    # Example usage of splitting data into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Example usage of Naive Bayes classifier
    #accuracy_nb = naive_bayes_classifier(X_train, X_test, y_train, y_test)
    #print("Naive Bayes Classifier Accuracy:", accuracy_nb)

    # Example usage of deep learning model
    #vocab_size = len(word_index) + 1  # Adding 1 to account for reserved 0 index
    #max_length = X.shape[1]
    #model = create_deep_learning_model(input_dim=vocab_size)
    #model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    #accuracy_dl = evaluate_model(model, X_test, y_test)
    #print("Deep Learning Model Accuracy:", accuracy_dl)

    # Additional Visualizations
    #generate_word_frequency_histogram(text)
    #generate_bar_chart_common_words(text)

if __name__ == "__main__":
    main()
