import tensorflow as tf
import keras
# from keras.optimizers
from encoder_decoder_withattn_architecture import Encoder, Decoder, TranslatorTrainer
from data import train_encoder_inputs, train_decoder_inputs, train_decoder_targets
from Utils.Text_Preprocess import loss_func

# parameters
source_vocab_size = 38539
target_vocab_size = 10556
hidden_dim = 256
embedding_dim = 128
epochs = 12
batch_size = 32

# dataset = tf.data.Dataset.from_tensor_slices((train_encoder_inputs, 
#                                               train_decoder_inputs, 
#                                               train_decoder_targets)).batch(batch_size, drop_remainder=True)

encoder = Encoder(source_vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(target_vocab_size, embedding_dim, hidden_dim)
optimizer = keras.optimizers.Adam()

translator_trainer = TranslatorTrainer(encoder, decoder)
# translator_trainer.compile(optimizer=optimizer, loss=loss_func)
# translator_trainer.fit(dataset, epochs=epochs)

# encoder.save_weights('attention_encoder_weights_with_dropout_ckpt')
# decoder.save_weights('attention_decoder_weights_with_dropout_ckpt')


