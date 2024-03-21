import os
import json
import keras
import numpy as np
import random
import pandas as pd
from encoder_decoder_wth_attn import Encoder_Decoder_Model
from data import preprocess_sentence ,max_decoder_seq_len, max_encoder_seq_len, source_tokenizer, target_tokenizer, target_vocab_size, source_vocab_size, SEPARATOR

file_path_test = os.path.join('Language_Translation_Using_EnD_withnoattnnodel', 'hun_eng_pairs_test.txt')

with open(file_path_test, 'r', encoding='utf-8') as file:
    test = [line.rstrip() for line in file]

# tokenizers path
source_file_path = os.path.join('Language_Translation_Using_EnD_withnoattnnodel', 'source_tokenizer.json')
target_file_path = os.path.join('Language_Translation_Using_EnD_withnoattnnodel', 'target_tokenizer.json')

with open(source_file_path) as f:
    data = json.load(f)
    source_tokenizer = keras.preprocessing.text.tokenizer_from_json(data)

with open(target_file_path) as f:
    data = json.load(f)
    target_tokenizer = keras.preprocessing.text.tokenizer_from_json(data)

model = Encoder_Decoder_Model()
model.load(name='Language_Translation_Using_EnD_withnoattnnodel\hun_eng_s2s_nmt_no_attention')
encoder = model.make_encoder_for_prediction()
decoder = model.make_decoder_for_prediction()

def translate_without_attention(sentence: str, 
                                source_tokenizer, encoder,
                                target_tokenizer, decoder,
                                max_translated_len = 30):

  # Vectorize the source sentence and run it through the encoder.    
  input_seq = source_tokenizer.texts_to_sequences([sentence])

  # Get the tokenized sentence to see if there are any unknown tokens.
  tokenized_sentence = source_tokenizer.sequences_to_texts(input_seq)

  states = encoder.predict(input_seq)  

  current_word = '<sos>'
  decoded_sentence = []

  while len(decoded_sentence) < max_translated_len:
    
    # Set the next input word for the decoder.
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = target_tokenizer.word_index[current_word]
    
    # Determine the next word.
    target_y_proba, h, c = decoder.predict([target_seq] + states)
    target_token_index = np.argmax(target_y_proba[0, -1, :])
    current_word = target_tokenizer.index_word[target_token_index]

    if (current_word == '<eos>'):
      break

    decoded_sentence.append(current_word)
    states = [h, c]
  
  return tokenized_sentence[0], ' '.join(decoded_sentence)

random.seed(1)
sentences = random.sample(test, 15)
sentences

def translate_sentences(sentences, translation_func, source_tokenizer, encoder,
                        target_tokenizer, decoder):
  translations = {'Tokenized Original': [], 'Reference': [], 'Translation': []}

  for s in sentences:
    source, target = s.split(SEPARATOR)
    source = preprocess_sentence(source)
    tokenized_sentence, translated = translation_func(source, source_tokenizer, encoder,
                                                      target_tokenizer, decoder)

    translations['Tokenized Original'].append(tokenized_sentence)
    translations['Reference'].append(target)
    translations['Translation'].append(translated)
  
  return translations

translations_no_attention = pd.DataFrame(translate_sentences(sentences, translate_without_attention,
                                                             source_tokenizer, encoder,
                                                             target_tokenizer, decoder))
print(translations_no_attention)