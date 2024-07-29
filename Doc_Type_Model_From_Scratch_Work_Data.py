###########################################################################################################################################################################
# Part 1 : Import required packages
import os
import re
import warnings
warnings.filterwarnings("ignore")

import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img 
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.utils import class_weight
from PIL import Image, UnidentifiedImageError

###########################################################################################################################################################################
# Part 2 : Set Directories

# Define directories
pdf_dir = r'C:\Users\eve.mcaleer\OneDrive______\Documents\Projects\#DISSERTATION#\Data\PDF_Data'
excel_path = r'C:\Users\eve.mcaleer\OneDrive______\Documents\Projects\#DISSERTATION#\_____.xlsx'

model_path = "CNN_Doc_Type_5.keras"

###########################################################################################################################################################################
# Part 3 : Get data (pdfs, and catalogue metadata)

# Read the Excel file, starting from row 4 as header
sheets = pd.read_excel(excel_path, sheet_name=None, header=3)

# Combine all the sheets into a single dataframe
catalogue_df = pd.concat(sheets.values(), ignore_index=True)

# Define the columns of interest
columns_of_interest = ['Sharepoint File name', 'Document Type']

# Filter the catalogue to only include the columns of interest
catalogue_df = catalogue_df[columns_of_interest]

# Remove .pdf extension from 'Sharepoint File name'
catalogue_df['Sharepoint File name'] = catalogue_df['Sharepoint File name'].str.replace('.pdf', '', regex=False)

###########################################################################################################################################################################
# Part 4 : Normalise/Clean the document type names & create Datagen

# Normalize document types (take only the first word and convert to lowercase)
catalogue_df['Document Type'] = catalogue_df['Document Type'].str.split().str[0].str.lower()
# Remove special characters from document types
catalogue_df['Document Type'] = catalogue_df['Document Type'].str.replace(r'[^\w\s]', '', regex=True)

# Define a new simplified mapping as per your instructions
class_mapping = {
    'memo': 'correspondence',
    'memorandum': 'correspondence',
    'correspondense': 'correspondence',
    'agenda': 'admin',
    'meeting': 'admin',
    'contract': 'admin',
    'diary': 'admin',
    'user': 'admin',
    'training': 'admin',
    'form': 'admin',
    'questionnaire': 'admin',
    'handout': 'misc',
    'guide': 'misc',
    'guidance': 'misc',
    'newsletter': 'misc',
    'booklet': 'misc',
    'instruction': 'misc',
    'instructions': 'misc',
    'index': 'misc',
    'information': 'misc',
    'diagram': 'misc',
    'graphs': 'misc',
    'note': 'misc',
    'map': 'misc',
    'document': 'misc',
    'table': 'misc',
    'program': 'misc',
    'case': 'misc',
    'chapter': 'misc',
    'reporttxt': 'misc',
    'reportarticle': 'misc',
    'proposal': 'misc',
    'report': 'scientific_report',
    'research': 'scientific_report',
    'presentation': 'presentation',   
    'powerpoint': 'presentation',     
    'poster': 'presentation'  }        

# Apply the mapping to the Document Type column
catalogue_df['Document Type'] = catalogue_df['Document Type'].replace(class_mapping)

# Filter out entries not matching correspondence, misc, admin, or presentation
valid_types = ['correspondence', 'misc', 'admin', 'presentation', 'scientific_report']
catalogue_df = catalogue_df[catalogue_df['Document Type'].isin(valid_types)]

# Dictionary to count occurrences of each document type
document_type_count = {}

# Iterate over all PDFs in the directory
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]  # Get the filename without extension
        
        # Find matching document type in the catalogue
        match = catalogue_df[catalogue_df['Sharepoint File name'] == pdf_name]
        
        if not match.empty:
            doc_type = match.iloc[0]['Document Type']
            
            if pd.notna(doc_type):  # Ensure document type is not NaN
                # Open the PDF and iterate through each page
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        # Update document type count
                        if doc_type in document_type_count:
                            document_type_count[doc_type] += 1
                        else:
                            document_type_count[doc_type] = 1

# Create a DataFrame to display the document type counts
doc_type_df = pd.DataFrame(list(document_type_count.items()), columns=['Document Type', 'Count'])

# Sort the DataFrame by count in descending order
doc_type_df = doc_type_df.sort_values(by='Count', ascending=False)

# Display the table
print("\nDocument Type Counts:")
print(doc_type_df.to_string(index=False))

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

###########################################################################################################################################################################
# Part 5: Convert PDFS to Images 

# Function to convert PDFs to images and match with document type
def convert_pdfs_to_images(pdf_dir, catalogue_df):
    results = []
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_name = pdf_file.replace('.pdf', '')
            
            # Find the row in the catalogue that matches the PDF name
            match = catalogue_df[catalogue_df['Sharepoint File name'] == pdf_name]
            
            if not match.empty:
                doc_type = match['Document Type'].values[0]
                
                # Open the PDF file
                pdf_path = os.path.join(pdf_dir, pdf_file)
                
                with pdfplumber.open(pdf_path) as pdf_file:
                    # Iterate through each page in the PDF
                    for page_num in range(len(pdf_file.pages)):
                        page = pdf_file.pages[page_num]
                        
                        # Convert the page to an image
                        img = page.to_image(resolution=72)
                        img = img.original  # Get the PIL Image object
                        
                        img = img.convert("RGB")  # Ensure the image is in RGB format
                        # Convert PIL image to numpy array
                        img_array = np.array(img)
                        
                        results.append([pdf_name, doc_type, img_array])
    return results

# Convert PDFs to images and get the results
pdf_images = convert_pdfs_to_images(pdf_dir, catalogue_df)

# Example: Display the first few results
print("pdf images examples...\n")
for pdf_name, doc_type, img_array in pdf_images[:3]:
    print(f"PDF: {pdf_name}, Document Type: {doc_type}, Image Shape: {img_array.shape}")

###########################################################################################################################################################################
# Part 6 : Encoding & splitting the data

# Extract images and labels
#images = [item[2] for item in pdf_images]
labels = [item[1] for item in pdf_images]

#print(f"Loaded {len(images)} images with shape {images[0].shape}")

# Convert labels to one-hot encoding
label_index = {label: index for index, label in enumerate(set(labels))}
print(f"Label index: {label_index}")
labels = [label_index[label] for label in labels]
labels = tf.keras.utils.to_categorical(np.asarray(labels))
print(f"Label shape: {labels.shape}")

# Convert images and labels to numpy arrays
#images = np.array(images)
#labels = np.array(labels)

# Split data into training (80%) and testing (20%)
#x_train, X_test, Y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Split training data further into training (80%) and validation (20%)
#X_train, X_val, y_train, y_val = train_test_split(x_train, Y_train, test_size=0.2, random_state=42)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(pdf_images, labels, test_size=0.2, random_state=42)

# Split training data further into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Example to show the lengths
print(f"Train set: {len(X_train)}, {len(y_train)}")
print(f"Validation set: {len(X_val)}, {len(y_val)}")
print(f"Test set: {len(X_test)}, {len(y_test)}")

def resize_images(data, target_size):
    resized_images = []
    for item in data:
        img = item[2]  # Extract the image array from the list
        if isinstance(img, np.ndarray):
            img = array_to_img(img)
        img_resized = img.resize(target_size)
        img_array = img_to_array(img_resized)
        resized_images.append(img_array)
    return np.array(resized_images)

# Define the target image size
target_size = (150, 150)

# Resize images and convert to NumPy arrays
X_train_resized = resize_images(X_train, target_size)
X_val_resized = resize_images(X_val, target_size)
X_test_resized = resize_images(X_test, target_size)

# Print shapes to verify
print(f"X_train_resized shape: {X_train_resized.shape}, y_train shape: {y_train.shape}")
print(f"X_val_resized shape: {X_val_resized.shape}, y_val shape: {y_val.shape}")
print(f"X_test_resized shape: {X_test_resized.shape}, y_test shape: {y_test.shape}")


# Calculate class weights to handle imbalanced dataset
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels.argmax(axis=1)), y=labels.argmax(axis=1))
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

print(f"Class weights: {class_weights}")
###########################################################################################################################################################################
# Part 7 : Create the  model

# Define constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 16
EPOCHS = 50

#Define early stopping/learning rate callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Define the model
model = Sequential([
    Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
    Dense(len(label_index), activation='softmax')
])

# Compile the model with a different learning rate
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.005),
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


###########################################################################################################################################################################
# Part 8: Run the model 

print("Fitting the model...\n")
history = model.fit(datagen.flow(X_train_resized, y_train, batch_size=BATCH_SIZE), 
                    epochs=EPOCHS, 
                    validation_data=(X_val_resized, y_val),
                    class_weight=class_weights,
                    callbacks=[early_stopping, lr_scheduler])


###########################################################################################################################################################################
# Part 9: Evaluate the Model

print("Analysing results...\n")
# Choose metrics
results = model.evaluate(X_test_resized, y_test, batch_size=BATCH_SIZE)
loss = results[0]
accuracy = results[1]
precision = results[2]
recall = results[3]

print('Test loss:', loss)
print('Test accuracy:', accuracy)
print('Test precision:', precision)
print('Test recall:', recall)

# Get predictions and true labels
y_pred = model.predict(X_test_resized)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
report = classification_report(y_true, y_pred_classes, target_names=label_index.keys())
print(report)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Calculate MCC
overall_mcc = matthews_corrcoef(y_true, y_pred_classes)
print(f'Overall Model Matthews Correlation Coefficient: {overall_mcc}')

# Compute MCC for each class (one-vs-all)
num_classes = len(np.unique(y_true))
class_mcc = []

for i in range(num_classes):
    y_true_binary = (y_true == i).astype(int)
    y_pred_binary = (y_pred == i).astype(int)
    mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
    class_mcc.append(mcc)
    print(f'Class {i+1} MCC: {mcc}')

# Print the individual MCCs for each class
for i, mcc in enumerate(class_mcc):
    print(f'Class {i+1} MCC: {mcc}')

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_index.keys(), yticklabels=label_index.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()