import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
import json
import typing
from tqdm import tqdm

# This is used when the Request does not give the output and tells you the Request was denied
headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}

def DataForLanguageTranlation(url: str, language: str, output_Dir: str, PrintProgress=None):
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, 'html.parser')

    links = soup.find_all('a')
    file_links = [link['href'] for link in links if '.' in link['href']]

    for link in file_links:
        file_url = url + link

        output_file = os.path.join(output_Dir, link)
        
        if PrintProgress is not None:
            print(f"Downloading {file_url}")

        file_response = requests.get(file_url)
        if file_response.status_code == 404:
            print(f"Could not download {file_url}")
            continue
    
        with open(output_file, 'wb') as f:
            f.write(file_response.content)

        if PrintProgress is not None:
            print(f"Downloaded : {link}")



language = "en-es"
url = f"https://data.statmt.org/opus-100-corpus/v1.0/supervised/{language}/"
output_directory = f"dataset_for_language_translation/{language}"
os.makedirs(output_directory, exist_ok=True)

DataForLanguageTranlation(url=url,
                          language=language,
                          output_Dir=output_directory,
                          PrintProgress=1)



def read_files(path):
    with open(path, "r", encoding="utf-8") as f:
        en_train_dataset = f.read().split("\n")[:-1]
    return en_train_dataset


def preprocess_inputs(data_batch, label_batch):
    # Assuming you have the tokenizer and detokenizer available
    data_batch_tokens = tokenizer.texts_to_sequences(data_batch)
    label_batch_tokens = detokenizer.texts_to_sequences(label_batch)

    # Padding sequences
    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(data_batch_tokens, padding='post')
    decoder_input = tf.keras.preprocessing.sequence.pad_sequences(label_batch_tokens, padding='post')[:, :-1]
    decoder_output = tf.keras.preprocessing.sequence.pad_sequences(label_batch_tokens, padding='post')[:, 1:]

    return (encoder_input, decoder_input), decoder_output

# Assuming you have train_dataset and val_dataset as tf.data.Dataset objects

# Create a function to apply the preprocessing to each batch
def preprocess_dataset(data_batch, label_batch):
    return preprocess_inputs(data_batch.numpy().tolist(), label_batch.numpy().tolist())

# Apply the preprocessing function to the datasets
train_dataset = train_dataset.map(preprocess_dataset)
val_dataset = val_dataset.map(preprocess_dataset)

# Batch and shuffle the datasets
batch_size = 4
train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=10000)
val_dataset = val_dataset.batch(batch_size)

# Now, train_dataset and val_dataset are instances of tf.data.Dataset with preprocessing applied.