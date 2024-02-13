# Create TensorFlow Transformer Model
import tensorflow as tf
from model import Transformer
from config import ModelConfigs
from data import tokenizer, detokenizer

configs = ModelConfigs()




transformer = Transformer(
    num_layers=configs.num_layers,
    d_model=configs.d_model,
    num_heads=configs.num_heads,
    dff=configs.dff,
    input_vocab_size=len(tokenizer)+1,
    target_vocab_size=len(detokenizer)+1,
    dropout_rate=configs.dropout_rate,
    encoder_input_size=tokenizer.max_length,
    decoder_input_size=detokenizer.max_length
    )

transformer.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=configs.init_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Compile the model
transformer.compile(
    loss=MaskedLoss(),
    optimizer=optimizer,
    metrics=[MaskedAccuracy()],
    run_eagerly=False
    )