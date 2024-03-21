import io
import json
import os
import regex as re
import numpy as np
import unicodedata
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def normalize_unicode(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(s):
  s = normalize_unicode(s)
  s = re.sub(r"([?.!,¿])", r" \1 ", s)
  s = re.sub(r'[" "]+', " ", s)
  s = s.strip()
  return s

def tag_target_sentences(sentences):
  tagged_sentences = map(lambda s: (' ').join(['<sos>', s, '<eos>']), sentences)
  return list(tagged_sentences)

def generate_decoder_inputs_targets(sentences, tokenizer):
  seqs = tokenizer.texts_to_sequences(sentences)
  decoder_inputs = [s[:-1] for s in seqs] # Drop the last token in the sentence.
  decoder_targets = [s[1:] for s in seqs] 
  return decoder_inputs, decoder_targets

def preprocess_data(data, SEPARATOR, source_tokenizer, target_tokenizer, max_encoder_seq_len, max_decoder_seq_len):
    train_input, train_target = map(list, zip(*[pair.split(SEPARATOR) for pair in data]))

    proprocessed_inputs = [preprocess_sentence(s) for s in train_input]
    proprocessed_targets = [preprocess_sentence(s) for s in train_target]
    proprocessed_targets = tag_target_sentences(proprocessed_targets)

    encoder_inputs = source_tokenizer.texts_to_sequences(proprocessed_inputs)
    decoder_inputs, decoder_target = generate_decoder_inputs_targets(proprocessed_targets, target_tokenizer)

    train_encoder_inputs = pad_sequences(encoder_inputs, maxlen=max_encoder_seq_len, padding='post', truncating='post')
    train_decoder_inputs = pad_sequences(decoder_inputs, max_decoder_seq_len, padding='post', truncating='post')
    train_decoder_targets = pad_sequences(decoder_target, maxlen=max_decoder_seq_len, padding='post', truncating='post')

    return train_encoder_inputs, train_decoder_inputs, train_decoder_targets

file_path_train = os.path.join('Language_Translation_Using_EnD_withnoattnnodel', 'hun_eng_pairs_train.txt')
file_path_val = os.path.join('Language_Translation_Using_EnD_withnoattnnodel', 'hun_eng_pairs_val.txt')

with open(file_path_train, 'r', encoding='utf-8') as file:
    train = [line.rstrip() for line in file]


with open(file_path_val, 'r', encoding='utf-8') as file:
    val = [line.rstrip() for line in file]

# special tokens
SEPARATOR = '<sep>'
StartOfSeq = '<sos>'
EndOfSeq = '<eos>'

# sepreating train inputs and train targets
train_input, train_target = map(list, zip(*[pair.split(SEPARATOR) for pair in train]))

# some print statements
# print(train_input[:1])
# print(train_target[:1])

proprocessed_inputs = [preprocess_sentence(s) for s in train_input]
proprocessed_targets = [preprocess_sentence(s) for s in train_target]
proprocessed_targets = tag_target_sentences(proprocessed_targets)

# some print statements
# print(proprocessed_inputs[:1])
# print(proprocessed_targets[:1])

source_tokenizer = Tokenizer(oov_token='<unk>', filters='"#$%&()*+-/:;=@[\\]^_`{|}~\t\n')
target_tokenizer = Tokenizer(oov_token='<unk>', filters='"#$%&()*+-/:;=@[\\]^_`{|}~\t\n')

source_tokenizer.fit_on_texts(proprocessed_inputs)
target_tokenizer.fit_on_texts(proprocessed_targets)

encoder_inputs = source_tokenizer.texts_to_sequences(proprocessed_inputs)
decoder_inputs, decoder_target = generate_decoder_inputs_targets(proprocessed_targets, target_tokenizer)

source_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1
max_encoder_seq_len = max([len(s) for s in encoder_inputs])
max_decoder_seq_len = max([len(s) for s in decoder_inputs])

# some print statements
# print(encoder_inputs[:1])
# print(decoder_inputs[:1])
# print(decoder_target[:1])
# print(source_vocab_size)
# print(max_encoder_seq_len)
# print(target_vocab_size)
# print(max_decoder_seq_len)

train_encoder_inputs = pad_sequences(encoder_inputs, maxlen=max_encoder_seq_len, padding='post', truncating='pre')
train_decoder_inputs = pad_sequences(decoder_inputs, max_decoder_seq_len, padding='post', truncating='post')
train_decoder_targets = pad_sequences(decoder_target, maxlen=max_decoder_seq_len, padding='post', truncating='post')

val_encoder_inputs, val_decoder_inputs, val_decoder_targets = preprocess_data(val, SEPARATOR=SEPARATOR, source_tokenizer=source_tokenizer, target_tokenizer=target_tokenizer,
                                                                            max_encoder_seq_len=max_encoder_seq_len, max_decoder_seq_len=max_decoder_seq_len)


# some print statements
# print(train_encoder_inputs.shape)
# print(train_decoder_inputs.shape)
# print(train_decoder_targets.shape)

# print(val_encoder_inputs.shape)
# print(val_decoder_inputs.shape)
# print(val_decoder_targets.shape)

# i already saved both the tokenizers
# source_tokenizer_json = source_tokenizer.to_json()
# with io.open('source_tokenizer.json', 'w', encoding='utf-8') as f:
#   f.write(json.dumps(source_tokenizer_json, ensure_ascii=False))

# target_tokenizer_json = target_tokenizer.to_json()
# with io.open('target_tokenizer.json', 'w', encoding='utf-8') as f:
#   f.write(json.dumps(target_tokenizer_json, ensure_ascii=False))
