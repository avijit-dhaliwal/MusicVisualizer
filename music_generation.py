import tensorflow as tf
import numpy as np

class MusicGenerator:
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        self.model = self._build_model(vocab_size, embedding_dim, rnn_units)
    
    def _build_model(self, vocab_size, embedding_dim, rnn_units):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model
    
    def generate(self, start_string, num_generate=1000, temperature=1.0):
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        
        text_generated = []
        self.model.reset_states()
        
        for i in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0) / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])
        
        return start_string + ''.join(text_generated)
    
    def train(self, dataset, epochs):
        self.model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))
        history = self.model.fit(dataset, epochs=epochs)
        return history

# Note: You'll need to prepare a dataset of MIDI files and create char2idx and idx2char mappings before training and using this model