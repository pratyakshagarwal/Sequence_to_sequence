from keras import layers
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import EarlyStopping

configuration = {
    'source_vocab_size':38539,
    'emb_dim': 128,
    'hidden_dense':256,
    'dropout_rate':0.2,
    'target_vocab_size': 10556,
    'learning_rate':0.01,
    'batch_size':32,
    'epochs':30,
    'max_encoding_len': 37,
    'mac_decoding_len':34,
}

class Encoder_Decoder_Model():
    def __init__(self, optimizer=Adam(learning_rate=configuration['learning_rate']), loss_fn=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()]):
        self.encoder_inputs = layers.Input(shape=[None], name='encoder_inputs')
        self.decoder_inputs = layers.Input(shape=[None], name='decoder_inputs')

        self.encoder_embeddings = layers.Embedding(configuration['source_vocab_size'], configuration['emb_dim'], mask_zero=True, name='encoder_embeddings')
        self.decoder_embeddings = layers.Embedding(configuration['target_vocab_size'], configuration['emb_dim'], mask_zero=True, name='decoder_embeddings')

        self.encoder_lstm = layers.LSTM(configuration['hidden_dense'], return_state=True, dropout=configuration['dropout_rate'], activation='relu', kernel_initializer='he_uniform', name='encoder_lstm')
        self.decoder_lstm = layers.LSTM(configuration['hidden_dense'], return_state=True, return_sequences=True, dropout=configuration['dropout_rate'], activation='relu', kernel_initializer='he_uniform', name='decoder_lstm')

        self.decoder_dense = layers.Dense(configuration['target_vocab_size'], activation='softmax', name='decoder_dense')

        self.encoder_embedding_output = self.encoder_embeddings(self.encoder_inputs)
        encoder_outputs, state_h, state_c = self.encoder_lstm(self.encoder_embedding_output)
        self.encoder_states = (state_h, state_c)

        self.decoder_embedding_output = self.decoder_embeddings(self.decoder_inputs)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedding_output, initial_state=self.encoder_states)
        self.y_proba = self.decoder_dense(self.decoder_outputs)

        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.y_proba, name='encoder_decoder_model_wthout_attn')

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    def train_model(self, train_encoder_inputs, train_decoder_inputs, train_decoder_targets, val_encoder_inputs=None, val_decoder_inputs=None, val_decoder_targets=None, epochs=configuration['epochs'], batch_size=configuration['batch_size'], callbacks=[EarlyStopping()]):
        
        if val_encoder_inputs is None:
            history = self.model.fit([train_encoder_inputs, train_decoder_inputs], train_decoder_targets,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.2,
                       callbacks=callbacks)
            
        else:
                history = self.model.fit([train_encoder_inputs, train_decoder_inputs], train_decoder_targets,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=([val_encoder_inputs, val_decoder_inputs], val_decoder_targets),
                       callbacks=callbacks)
            
        return history
            
    def make_encoder_for_prediction(self):
        encoder_inputs = self.model.get_layer('encoder_inputs').input
        encoder_embeddings_layer = self.model.get_layer('encoder_embeddings')
        encoder_lstm = self.model.get_layer('encoder_lstm')

        encoder_embeddings = encoder_embeddings_layer(encoder_inputs)
        _, state_h, state_c = encoder_lstm(encoder_embeddings)
        output_encoder_state = [state_h, state_c]

        encoder = Model(encoder_inputs, output_encoder_state)
        return encoder

    def make_decoder_for_prediction(self):
        decoder_inputs = self.model.get_layer('decoder_inputs').input
        decoder_embedding_layer = self.model.get_layer('decoder_embeddings')
        decoder_lstm = self.model.get_layer('decoder_lstm')
        decoder_dense = self.model.get_layer('decoder_dense')

        decoder_input_state_h = layers.Input(shape=(configuration['hidden_dense'],), name='decoder_input_state_h')
        decoder_input_state_c = layers.Input(shape=(configuration['hidden_dense'],), name='decoder_input_state_c')
        decoder_input_states = [decoder_input_state_h, decoder_input_state_c]

        decoder_embeddings = decoder_embedding_layer(decoder_inputs)
        decoder_output_sequence, decoder_output_state_h, decoder_output_state_c = decoder_lstm(decoder_embeddings)
        decoder_output_states = [decoder_output_state_h, decoder_output_state_c]
        y_proba = decoder_dense(decoder_output_sequence)

        decoder = Model([decoder_inputs] + decoder_input_states,
                             [y_proba] + decoder_output_states)
        
        return decoder
        
    def translate(self, sentence, source_tokenizer, target_tokenizer, max_translated_len = 30):
        input_seq = source_tokenizer.texts_to_sequences([sentence])
        # print(input_seq.shape)
        tokenized_sentence = source_tokenizer.sequences_to_texts(input_seq)
        print(tokenized_sentence)

        encoder = self.make_encoder_for_prediction()
        decoder = self.make_decoder_for_prediction()
        states = encoder([input_seq])

        current_word = '<sos>'
        decoded_sentence = []

        while len(decoded_sentence) < max_translated_len:
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = target_tokenizer.word_index[current_word]

            # determine the next word
            target_y_proba, h, c = decoder.predict([target_seq] + states)
            target_token_index = np.argmax(target_y_proba[0, -1, :])
            current_word = target_tokenizer.index_word[target_token_index]

            if (current_word == '<eos>'):
                break

            decoded_sentence.append(current_word)
            states = [h, c]

            return tokenized_sentence[0], ' '.join(decoded_sentence)
        
    def plot_end_model(self, to_file='hun_eng_seq2seq_nmt_no_attention.png'):
        plot_model(self.model, to_file=to_file, show_shapes=True, show_layer_names=True)

    def plot_encoder_model(self, to_file='encoder_model_architecture.png'):
        plot_model(self.encoder, to_file=to_file, show_shapes=True, show_layer_names=True)

    def plot_decoder_model(self, to_file='decoder_model_architecture.png'):
        plot_model(self.decoder, to_file=to_file, show_shapes=True, show_layer_names=True)

    def save_model(self, name='hun_to_eng_withnoattn'):
        self.model.save(name)

    def load(self, name='hun_to_eng_withnoattn'):
        self.model = load_model(name)

    def get_summary(self):
        return self.model.summary()
    
if __name__ == '__main__':
    model = Encoder_Decoder_Model()
    print(model.get_summary())
    # model.plot_end_model()