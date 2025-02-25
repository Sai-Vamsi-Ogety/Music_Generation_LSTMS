import tensorflow as tf

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
	
	def LSTM(rnn_units): 
	  return tf.keras.layers.LSTM(
	    rnn_units, 
	    return_sequences=True, 
	    recurrent_initializer='glorot_uniform',
	    recurrent_activation='sigmoid',
	    stateful=True,
	  )

	model = tf.keras.Sequential([

	tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
	LSTM(rnn_units), 
	tf.keras.layers.Dense(vocab_size)
	])

	return model

if __name__ == "__main__" :
    build_model()


