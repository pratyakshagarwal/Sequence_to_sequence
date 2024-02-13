import os
from dataprepratation_for_Language_translation import DataForLanguageTranlation, preprocess_inputs, read_files, preprocess_dataset
from Tokenizer.tokenizer import CustomTokenizer
from config import ModelConfigs

configs = ModelConfigs()

language = "en-es"
url = f"https://data.statmt.org/opus-100-corpus/v1.0/supervised/{language}/"
output_directory = f"dataset_for_language_translation/{language}"
os.makedirs(output_directory, exist_ok=True)

DataForLanguageTranlation(url=url,
                          language=language,
                          output_Dir=output_directory,
                          PrintProgress=1)


# Path to dataset
en_training_data_path = "dataset_for_language_translation/en-es/opus.en-es-train.en"
en_validation_data_path = "dataset_for_language_translation/en-es/opus.en-es-dev.en"
es_training_data_path = "dataset_for_language_translation/en-es/opus.en-es-train.es"
es_validation_data_path = "dataset_for_language_translation/en-es/opus.en-es-dev.es"


en_training_data = read_files(en_training_data_path)
en_validation_data = read_files(en_validation_data_path)
es_training_data = read_files(es_training_data_path)
es_validation_data = read_files(es_validation_data_path)

# Consider only sentences with length <= 500
max_lenght = 500
train_dataset = [[es_sentence, en_sentence] for es_sentence, en_sentence in zip(es_training_data, en_training_data) if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght]
val_dataset = [[es_sentence, en_sentence] for es_sentence, en_sentence in zip(es_validation_data, en_validation_data) if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght]
es_training_data, en_training_data = zip(*train_dataset)
es_validation_data, en_validation_data = zip(*val_dataset)


# prepare spanish tokenizer, this is the input language
tokenizer = CustomTokenizer(char_level=True)
tokenizer.fit_on_texts(es_training_data)
tokenizer.save(configs.model_path + "/tokenizer.json")

# prepare english tokenizer, this is the output language
detokenizer = CustomTokenizer(char_level=True)
detokenizer.fit_on_texts(en_training_data)
detokenizer.save(configs.model_path + "/detokenizer.json")