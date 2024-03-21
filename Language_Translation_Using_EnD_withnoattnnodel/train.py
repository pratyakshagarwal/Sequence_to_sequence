from keras.callbacks import ModelCheckpoint, EarlyStopping
from encoder_decoder_wth_attn import Encoder_Decoder_Model
from data import train_encoder_inputs, train_decoder_inputs, train_decoder_targets, val_encoder_inputs, val_decoder_inputs, val_decoder_targets, max_decoder_seq_len, max_encoder_seq_len, source_tokenizer, target_tokenizer, target_vocab_size, source_vocab_size

if __name__ == '__main__':
    # Saving this to a folder on my local machine.
    filepath=r"./Language_Translation_Using_EnD_withnoattnnode/training1/cp.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=filepath, save_weights_only=True, verbose=1)
    es_callback = EarlyStopping(monitor='val_loss', patience=3)

    model = Encoder_Decoder_Model()
    history = model.train_model(train_encoder_inputs=train_encoder_inputs,
                                train_decoder_inputs=train_decoder_inputs,
                                train_decoder_targets=train_decoder_targets,
                                val_encoder_inputs=val_encoder_inputs,
                                val_decoder_inputs=val_decoder_inputs,
                                val_decoder_targets=val_decoder_targets,
                                callbacks=[cp_callback, es_callback])
    
    model.save_model()

    