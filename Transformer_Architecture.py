import tensorflow as tf
import numpy as np
from tensorflow import keras

# Importing custom masking functions
from Transformers_parts.masking import get_encoder_self_attention_mask,get_decoder_self_attention_mask, get_encoder_decoder_attention_mask


# Function for scaled dot-product attention
def scaled_dot_product_attention(query, key, value, mask=None):
  key_dim = tf.cast(tf.shape(key)[-1], tf.float16)
  scaled_scores = tf.matmul(query, key, transpose_b=True) / np.sqrt(key_dim)

  if mask is not None:
    scaled_scores = tf.where(mask==0, -np.inf, scaled_scores)

  softmax = tf.keras.layers.Softmax()
  weights = softmax(scaled_scores) 
  return tf.matmul(weights, value), weights


# Multi-head self-attention layer
class MultiHeadSelfAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadSelfAttention, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_head = self.d_model // self.num_heads

    # Linear transformations for query, key, and value
    self.wq = tf.keras.layers.Dense(self.d_model)
    self.wk = tf.keras.layers.Dense(self.d_model)
    self.wv = tf.keras.layers.Dense(self.d_model)

    # Linear layer to generate the final output.
    self.dense = tf.keras.layers.Dense(self.d_model)
  
  def split_heads(self, x):
    batch_size = x.shape[0]

    split_inputs = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
    return tf.transpose(split_inputs, perm=[0, 2, 1, 3])
  
  def merge_heads(self, x):
    batch_size = x.shape[0]

    merged_inputs = tf.transpose(x, perm=[0, 2, 1, 3])
    return tf.reshape(merged_inputs, (batch_size, -1, self.d_model))

  def call(self, q, k, v, mask):
    qs = self.wq(q)
    ks = self.wk(k)
    vs = self.wv(v)

    qs = self.split_heads(qs)
    ks = self.split_heads(ks)
    vs = self.split_heads(vs)

    output, attn_weights = scaled_dot_product_attention(qs, ks, vs, mask)
    output = self.merge_heads(output)

    return self.dense(output), attn_weights


# Function for the feed-forward network
def feed_forward_network(d_model, hidden_dim):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(hidden_dim, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])


# Encoder block
class EncoderBlock(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
    super(EncoderBlock, self).__init__()

    self.mhsa = MultiHeadSelfAttention(d_model, num_heads)
    self.ffn = feed_forward_network(d_model, hidden_dim)

    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    self.layernorm1 = tf.keras.layers.LayerNormalization()
    self.layernorm2 = tf.keras.layers.LayerNormalization()
  
  def call(self, x, training, mask):
    mhsa_output, attn_weights = self.mhsa(x, x, x, mask)
    mhsa_output = self.dropout1(mhsa_output, training=training)
    mhsa_output = self.layernorm1(x + mhsa_output)

    ffn_output = self.ffn(mhsa_output)
    ffn_output = self.dropout2(ffn_output, training=training)
    output = self.layernorm2(mhsa_output + ffn_output)

    return output, attn_weights


# Encoder layer  
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_blocks, d_model, num_heads, hidden_dim, src_vocab_size,
               max_seq_len, dropout_rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.max_seq_len = max_seq_len

    self.token_embed = tf.keras.layers.Embedding(src_vocab_size, self.d_model)
    self.pos_embed = tf.keras.layers.Embedding(max_seq_len, self.d_model)

    # The original Attention Is All You Need paper applied dropout to the
    # input before feeding it to the first encoder block.
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # Create encoder blocks.
    self.blocks = [EncoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate) 
    for _ in range(num_blocks)]
  
  def call(self, input, training, mask):
    token_embeds = self.token_embed(input)

    # Generate position indices for a batch of input sequences.
    num_pos = input.shape[0] * self.max_seq_len
    pos_idx = np.resize(np.arange(self.max_seq_len), num_pos)
    pos_idx = np.reshape(pos_idx, input.shape)
    pos_embeds = self.pos_embed(pos_idx)

    x = self.dropout(token_embeds + pos_embeds, training=training)

    # Run input through successive encoder blocks.
    for block in self.blocks:
      x, weights = block(x, training, mask)

    return x, weights
  

# Decoder block
class DecoderBlock(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
    super(DecoderBlock, self).__init__()

    self.mhsa1 = MultiHeadSelfAttention(d_model, num_heads)
    self.mhsa2 = MultiHeadSelfAttention(d_model, num_heads)

    self.ffn = feed_forward_network(d_model, hidden_dim)

    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    self.layernorm1 = tf.keras.layers.LayerNormalization()
    self.layernorm2 = tf.keras.layers.LayerNormalization()
    self.layernorm3 = tf.keras.layers.LayerNormalization()
  
  # Note the decoder block takes two masks. One for the first MHSA, another
  # for the second MHSA.
  def call(self, encoder_output, target, training, decoder_mask, memory_mask):
    mhsa_output1, attn_weights = self.mhsa1(target, target, target, decoder_mask)
    mhsa_output1 = self.dropout1(mhsa_output1, training=training)
    mhsa_output1 = self.layernorm1(mhsa_output1 + target)

    mhsa_output2, attn_weights = self.mhsa2(mhsa_output1, encoder_output, 
                                            encoder_output, 
                                            memory_mask)
    mhsa_output2 = self.dropout2(mhsa_output2, training=training)
    mhsa_output2 = self.layernorm2(mhsa_output2 + mhsa_output1)

    ffn_output = self.ffn(mhsa_output2)
    ffn_output = self.dropout3(ffn_output, training=training)
    output = self.layernorm3(ffn_output + mhsa_output2)

    return output, attn_weights

# Decoder layer
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
               max_seq_len, dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.max_seq_len = max_seq_len

    # Token and positional embeddings
    self.token_embed = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
    self.pos_embed = tf.keras.layers.Embedding(max_seq_len, self.d_model)

    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    self.blocks = [DecoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate) for _ in range(num_blocks)]

  def call(self, encoder_output, target, training, decoder_mask, memory_mask):
    token_embeds = self.token_embed(target)

    # Generate position indices.
    num_pos = target.shape[0] * self.max_seq_len
    pos_idx = np.resize(np.arange(self.max_seq_len), num_pos)
    pos_idx = np.reshape(pos_idx, target.shape)

    pos_embeds = self.pos_embed(pos_idx)

    x = self.dropout(token_embeds + pos_embeds, training=training)

    for block in self.blocks:
      x, weights = block(encoder_output, x, training, decoder_mask, memory_mask)

    return x, weights
  

# Transformer model
class Transformer(tf.keras.Model):
  def __init__(self, num_blocks, d_model, num_heads, hidden_dim, source_vocab_size,
               target_vocab_size, max_input_len, max_target_len, dropout_rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_blocks, d_model, num_heads, hidden_dim, source_vocab_size, 
                           max_input_len, dropout_rate)
    
    self.decoder = Decoder(num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
                           max_target_len, dropout_rate)
    
    # The final dense layer to generate logits from the decoder output.
    self.output_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, input_seqs, target_input_seqs, training, encoder_mask,
           decoder_mask, memory_mask):
    encoder_output, encoder_attn_weights = self.encoder(input_seqs, 
                                                        training, encoder_mask)

    decoder_output, decoder_attn_weights = self.decoder(encoder_output, 
                                                        target_input_seqs, training,
                                                        decoder_mask, memory_mask)

    return self.output_layer(decoder_output), encoder_attn_weights, decoder_attn_weights

## Tuneable Parameters
batch_size=8
d_model = 512
num_blocks = 6
hidden_dim = 2048
num_heads = 8
dropout_rate = 0.1

# Untunable Parameters
source_vocab_size = 100
target_vocab_size = 110
max_target_len = 10
max_input_length = 11


# Masking for attention mechanisms
encoder_mask = get_encoder_self_attention_mask(seq_len=max_input_length)
decoder_mask = get_decoder_self_attention_mask(seq_len=max_target_len)
memory_mask = get_encoder_decoder_attention_mask(encoder_seq_len=max_input_length, decoder_seq_len=max_target_len)


# Function to generate random sequences for testing
def generate_random_sequences(vocab_size, sequence_length, num_sequences):
    return np.random.randint(0, vocab_size, size=(num_sequences, sequence_length))

# Generate random input sequences
input_seqs = generate_random_sequences(source_vocab_size, max_input_length, num_sequences=batch_size)
target_input_seqs = generate_random_sequences(target_vocab_size, max_target_len, num_sequences=batch_size)


# Instantiate the Transformer model
transformer_model = Transformer(
    num_blocks=num_blocks,
    d_model=d_model,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    source_vocab_size=source_vocab_size,
    target_vocab_size=target_vocab_size,
    max_input_len=max_input_length,
    max_target_len=max_target_len,
    dropout_rate=dropout_rate
)

# Call the Transformer model
output, encoder_attn_weights, decoder_attn_weights = transformer_model(
    input_seqs, target_input_seqs, training=True,
    encoder_mask=encoder_mask, decoder_mask=decoder_mask, memory_mask=memory_mask
)

# Print the output shape and attention weights shapes
print("input_seq",input_seqs.shape)
print("target_input_Seq", target_input_seqs.shape)
print("Output Shape:", output.shape)
print("Encoder Attention Weights Shape:", encoder_attn_weights.shape)
print("Decoder Attention Weights Shape:", decoder_attn_weights.shape)