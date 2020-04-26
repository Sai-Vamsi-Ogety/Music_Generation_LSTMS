from dataset import vocab, char2idx, idx2char, songs_joined
import numpy as np

if __name__ == "__main__" :

    global vectorized_songs

    vectorized_songs = np.array([char2idx[x] for x in songs_joined])

    print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
    np.savetxt('../input/vectorized_input.txt',vectorized_songs)
	# check that vectorized_songs is a numpy array
	# if type(vectorized_songs) != np.ndarray:

	#     print("returned result should be a numpy array")