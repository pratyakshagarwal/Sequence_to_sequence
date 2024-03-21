import json
import os
import keras
import random
import numpy as np
import pandas as pd
from train import encoder, decoder
from keras.preprocessing.sequence import pad_sequences
from data import max_encoder_seq_len, preprocess_sentence, SEPARATOR


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


encoder.load_weights(r'Language_Translation_Using_End_withattn\attention_weights\attention_decoder_weights_ckpt.data-00000-of-00001')
decoder.load_weights(r'Language_Translation_Using_End_withattn\attention_weights\attention_decoder_weights_ckpt.data-00000-of-00001')


def translate_with_attention(sentence: str, 
                             source_tokenizer, encoder,
                             target_tokenizer, decoder,
                             max_translated_len = 30):
    input_seq = source_tokenizer.texts_to_sequences([sentence])
    tokenized = source_tokenizer.sequences_to_texts(input_seq)

    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_len, padding='post')
    encoder_output, state_h, state_c  = encoder.predict(input_seq)

    current_word = '<sos>'
    decoded_sentence = []

    while len(decoded_sentence) < max_translated_len:
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = target_tokenizer.word_index[current_word]

        logits, state_h, state_c, _ = decoder.predict([target_seq, encoder_output, (state_h, state_c)])
        current_token_index = np.argmax(logits[0])

        current_word = target_tokenizer.index_word[current_token_index]

        if (current_word == '<eos>'):
          break

        decoded_sentence.append(current_word)

    return tokenized[0], ' '.join(decoded_sentence)


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


random.seed(1)
sentences = random.sample(test, 15)
sentences

translations_w_attention = pd.DataFrame(translate_sentences(sentences, translate_with_attention,
                                                                    source_tokenizer, encoder,
                                                                    target_tokenizer, decoder))

print(translations_w_attention)