import tensorflow as tf 

def compute_loss(labels, logits):
	
	loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  
	return loss
