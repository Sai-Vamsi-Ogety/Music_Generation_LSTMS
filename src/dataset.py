# Download and import the MIT 6.S191 package
import mitdeeplearning as mdl
import numpy as np
import json

songs = mdl.lab1.load_training_data()

# Print one of the songs to inspect it in greater detail!
example_song = songs[0]
print("\nExample song: ")
print(example_song)

# Join our list of song strings into a single string containing all songs
global songs_joined
songs_joined = "\n\n".join(songs) 

# Find all unique characters in the joined string
global vocab
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")
# create two lookup tables
# 1. text to integers
global char2idx
char2idx = {u:i for i, u in enumerate(vocab)}
# json.dump(char2idx, open("../input/char2idx.json",'w'))
# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
global idx2char
idx2char = np.array(vocab)
# np.savetxt('../input/idx2char.txt', idx2char, fmt='%s' )

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

