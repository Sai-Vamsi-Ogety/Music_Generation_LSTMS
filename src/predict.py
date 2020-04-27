# To keep this inference step simple, we will use a batch size of 1. Because of how the RNN state is passed from timestep to timestep, 
# the model will only be able to accept a fixed batch size once it is built.

# To run the model with a different batch_size, we'll need to rebuild the model and restore the weights from the latest checkpoint, 
# i.e., the weights after the last checkpoint during training:
from dataset import char2idx, idx2char,vocab
from IPython import display as ipythondisplay
from model import build_model
from tqdm import tqdm
import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np 

vocab_size = 83
embedding_dim = 256 
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = './training_checkpoints_2000_iterations'


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1) # TODO

# Restore the model weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_string, generation_length=1000):
    # Evaluation step (generating ABC text using the learned RNN model)

    '''TODO: convert the start string to numbers (vectorize)'''
    input_eval = [char2idx[s] for s in start_string] # TODO
    # input_eval = ['''TODO''']
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    #tqdm._instances.clear()

    for i in range(generation_length):
        '''TODO: evaluate the inputs and generate the next character predictions'''
        predictions = model(input_eval)
        # predictions = model('''TODO''')

        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        '''TODO: use a multinomial distribution to sample'''
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        # predicted_id = tf.random.categorical('''TODO''', num_samples=1)[-1,0].numpy()

        # Pass the prediction along with the previous hidden state
        #   as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        '''TODO: add the predicted character to the generated text!'''
        # Hint: consider what format the prediction is in vs. the output
        text_generated.append(idx2char[predicted_id]) # TODO 
        # text_generated.append('''TODO''')

    return (start_string + ''.join(text_generated))

generated_songs = generate_text(model, start_string="X", generation_length=1000)
print(generated_songs[0])
file1 = open("/home/jovyan/work/input/generated_songs_2000.txt","w") 
file1.write(generated_songs)
file1.close() 

# for i, song in enumerate(generated_songs): 
    
#     # Synthesize the waveform from a song
#     waveform = mdl.lab1.play_song(song)
#     def save_song_to_abc(song,filename = "tmp")
#         save_name = "../input/{}.abc".format(tmp)
#         with open(save_name, "w") as f:
#             f.write(song)
#         return tmp
    


#     # If its a valid song (correct syntax), lets play it! 
#     if waveform:
#         print("Generated song", i)
#         ipythondisplay.display(waveform)
