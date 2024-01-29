import tensorflow as tf

def get_encoder_self_attention_mask(seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask  # 1's in lower triangle, 0's in upper triangle

def get_decoder_self_attention_mask(seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), 0, -1)
    return mask


def get_encoder_decoder_attention_mask(encoder_seq_len, decoder_seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((decoder_seq_len, encoder_seq_len)), -1, 0)
    return mask  # 1's in lower triangle, 0's in upper triangle
