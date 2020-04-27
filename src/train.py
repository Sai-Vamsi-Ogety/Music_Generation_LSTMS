from  get_batch import create_batch
from dataset import char2idx, idx2char,vocab
import numpy as np 
import json
from model import build_model
from loss import compute_loss
import os 
import tensorflow as tf 
import mitdeeplearning as mdl
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm



#get the vectorized input
vectorized_songs = np.loadtxt('../input/vectorized_input.txt')

# split into batches
x_batch, y_batch = create_batch(vectorized_songs, seq_length=5, batch_size=1)

#Hyperparameters
# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 32  # Experiment between 1 and 64
seq_length = 200  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters: 
vocab_size = len(vocab)
embedding_dim = 256 
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = './training_checkpoints_2000_iterations'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")


# Build the model
model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_step(x, y): 
  # Use tf.GradientTape()
  with tf.GradientTape() as tape:


    y_hat = model(x) 

    loss = compute_loss(y, y_hat) 

    grads = tape.gradient(loss, model.trainable_variables) 

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

##################
# Begin training!#
##################

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for iter in tqdm(range(num_training_iterations)):

  # Grab a batch and propagate it through the network
  x_batch, y_batch = create_batch(vectorized_songs, seq_length, batch_size)
  loss = train_step(x_batch, y_batch)

  # Update the progress bar
  history.append(loss.numpy().mean())
  plotter.plot(history)

  # Update the model with the changed weights!
  if iter % 100 == 0:     
    model.save_weights(checkpoint_prefix)
    
# Save the trained model and the weights
model.save_weights(checkpoint_prefix)

