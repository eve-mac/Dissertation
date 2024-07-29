###########################################################################################################################################################################
# Part 1 : Import required packages
import re
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup 
import nltk
import pdfplumber

from nltk.corpus import stopwords
import tensorflow as tf
from transformers import BartTokenizer, TFAutoModelForSeq2SeqLM

# Set maximum column width to display more text
pd.set_option("display.max_colwidth", 200)

###########################################################################################################################################################################
# Part 2 : Define paths

# Define directories
pdf_dir = r'C:\Users\eve.mcaleer\OneDrive______\Documents\Projects\#DISSERTATION#\Data\PDF_Data'
excel_path = r'C:\Users\eve.mcaleer\OneDrive______\Documents\Projects\#DISSERTATION#\_____.xlsx'

# Defining the path to the saved models directory where the ZIP files are to be extracted
saved_models_dir = r'C:\Users\eve.mcaleer\OneDrive - Forest Research\Documents\Projects\#DISSERTATION#\Code\text_summary' 



###########################################################################################################################################################################
# Part 3 : Reading in the data

# Read the Excel file, starting from row 4 as header
print('Reading in catalogue...\n')
sheets = pd.read_excel(excel_path, sheet_name=None, header=3)

# Combine all the sheets into a single dataframe
catalogue_df = pd.concat(sheets.values(), ignore_index=True)

# Define the columns of interest
columns_of_interest = ['Sharepoint File name', 'Document Type', 'Description']

# Filter the catalogue to only include the columns of interest
print('Gathering columns of interest...\n')
catalogue_df = catalogue_df[columns_of_interest]

# Ensure 'Sharepoint File name' column does not have .pdf extension
catalogue_df['Sharepoint File name'] = catalogue_df['Sharepoint File name'].str.replace('.pdf', '', regex=False)

# Normalize document types (take only the first word and convert to lowercase, remove special characters)
#catalogue_df['Document Type'] = catalogue_df['Document Type'].str.split().str[0].str.lower()
#catalogue_df['Document Type'] = catalogue_df['Document Type'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Create an empty list to store the results
results = []

print('Iterating through pdfs...\n')
# Iterate over all PDFs in the directory
for pdf in os.listdir(pdf_dir):
    if pdf.endswith('.pdf'):
        pdf_name = pdf.replace('.pdf', '')
        
        # Find the row in the catalogue that matches the PDF name
        match = catalogue_df[catalogue_df['Sharepoint File name'] == pdf_name]
        
        if not match.empty:
            doc_type = match['Document Type'].values[0]
            subject = match['Description'].values[0]
            
            # Open the PDF file
            pdf_path = os.path.join(pdf_dir, pdf)
            
            with pdfplumber.open(pdf_path) as pdf_file:
                # Store the text for each page
                pdf_text = ""
                
                # Iterate through each page in the PDF
                for page_num in range(len(pdf_file.pages)):
                    page = pdf_file.pages[page_num]
                    
                    # Extract the text from the page
                    text = page.extract_text()
                    if text:
                        pdf_text += text
                
                results.append([pdf, doc_type, subject, pdf_text])

# Create a DataFrame from the results
data = pd.DataFrame(results, columns=['PDF Name', 'Document Type', 'Description', 'Text'])

print("Metadata and text extraction complete. Data stored in DataFrame. \n")

###########################################################################################################################################################################
# Part 4 : Text Preprocessing

# Print column names
print("Column names before dropping duplicates and nulls...\n:")
print(data.columns)

# Dropping Duplicates and nulls

# Print the number of rows
print("Number of rows:", data.shape[0]) #   rows

data.drop_duplicates(subset=['Text'], inplace=True)  # dropping duplicates
data.dropna(axis=0, inplace=True)   # dropping na

print("Column names after dropping duplicates and nulls...\n:")
print("Number of rows:", data.shape[0]) #   rows 

# Contraction mapping dictionary
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

# Cleaning the text 

print("Printing 5 rows of pdf text...\n")
print(data['Text'].iloc[1:7])

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

# text preprocessing
def text_cleaner(data_text):
    # Change variable name newString to data_text
    data_text = data_text.lower()
    data_text = BeautifulSoup(data_text, "lxml").text
    data_text = re.sub(r'\([^)]*\)', '', data_text)
    data_text = re.sub('"', '', data_text)
    data_text = ' '.join([contraction_mapping[a] if a in contraction_mapping else a for a in data_text.split(" ")])
    data_text = re.sub(r"'s\b", "", data_text)
    data_text = re.sub("[^a-zA-Z]", " ", data_text)
    tokens = [b for b in data_text.split() if not b in stop_words]
    long_words = []
    for c in tokens:
        if len(c) >= 3:  # removing short words
            long_words.append(c)
    return (" ".join(long_words)).strip()

cleaned_text = []
for d in data['Text']:
    cleaned_text.append(text_cleaner(d))

# Cleaning the descriptions (summaries)

print("Printing 5 rows of descriptions...\n")
print(data['Description'].iloc[1:7])

# Summary preprocessing
def summary_cleaner(desc):
    data_summary = re.sub('"', '', desc)
    data_summary = ' '.join([contraction_mapping[b] if b in contraction_mapping else b for b in data_summary.split(" ")])    
    data_summary = re.sub(r"'s\b", "", data_summary)
    data_summary = re.sub("[^a-zA-Z]", " ", data_summary)
    data_summary = data_summary.lower()
    tokens = data_summary.split()
    data_summary = ''
    for a in tokens:
        if len(a) > 3:                                 
            data_summary = data_summary + a + ' '  
    return data_summary.strip()

# Call the above function
cleaned_summary = []
for b in data['Description']:
    cleaned_summary.append(summary_cleaner(b))

print("Adding cleaned data to dataframe...\n")
data['cleaned_text'] = cleaned_text
data['cleaned_summary'] = cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)

print(data.columns)

# Define the range of rows you want to display (2 to 7 for example)
row_range = range(2, 8)

# Print original text aligned
print("Printing original text...\n")
for i in row_range:
    original_text = data['Text'].iloc[i][:500] # limit to first 500 characters
    print(f"{i}{' '*(5-len(str(i)))} {original_text}\n")

# Print cleaned text aligned
print("Printing cleaned text...\n")
for i in row_range:
    cleaned_text = data['cleaned_text'].iloc[i][:500]
    print(f"{i}{' '*(5-len(str(i)))} {cleaned_text}\n")

# Print original description aligned
print("Printing original description...\n")
for i in row_range:
    original_description = data['Description'].iloc[i][:500]
    print(f"{i}{' '*(5-len(str(i)))} {original_description}\n")

# Print cleaned description aligned
print("Printing cleaned description...\n")
for i in row_range:
    cleaned_description = data['cleaned_summary'].iloc[i][:500]
    print(f"{i}{' '*(5-len(str(i)))} {cleaned_description}\n")

###########################################################################################################################################################################
# Part 5 : Model loading and Printing Results

print("Extracting model...\n")

# Define paths to the extracted folders
bart_model_dir = os.path.join(saved_models_dir, 'bart_model')
bart_tokenizer_dir = os.path.join(saved_models_dir, 'bart_tokenizer')

# Function to load pretrained model and tokenizer
def load_bart_model_and_tokenizer(model_dir, tokenizer_dir):
    # Load the tokenizer and model from the extracted directories
    tokenizer = BartTokenizer.from_pretrained(tokenizer_dir)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model

# Load tokenizer and model
print("Loading model...\n")
tokenizer, model = load_bart_model_and_tokenizer(bart_model_dir, bart_tokenizer_dir)

# Assuming 'data' is your DataFrame already loaded with cleaned text and summaries
texts = data['cleaned_text'][:15]
real_summaries = data['cleaned_summary'][:15]

# Iterate through the DataFrame rows
for i, row in data.iterrows():
    text = row['cleaned_text']
    
    # Tokenize and generate summary
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    input_ids = inputs['input_ids']

    # Ensure input_ids and attention_mask are within expected range
    input_ids = input_ids[:, :1024]  # Truncate to maximum supported length by BART model
    attention_mask = inputs['attention_mask'][:, :1024] if 'attention_mask' in inputs else None
    
    # Generate summary
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=20,           # Increase the number of beams
        max_length=15,          # Increase the maximum length of the summary
        early_stopping=True,    # Enable early stopping
        length_penalty=2.5      # Adjust length penalty if needed
    )
    summary = tokenizer.decode(summary_ids.numpy()[0], skip_special_tokens=True)

    # Print summaries
    print("\nReal Summary:")
    if i in data.index:
        print(data.loc[i, 'cleaned_summary'])
    else:
        print("Summary not found for index:", i)

    print("\nBART-generated Summary:")
    print(summary)
    print("\n--------------------------------------\n")
