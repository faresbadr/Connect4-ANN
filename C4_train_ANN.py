"""

Script that creates a keras model, and trains it with given data.
Loads data from training set containing 'keys', 'values', and 'moves'
    - Key:
        Number that represents a unique board state
    - value:
        Number that represents if a board state is a win or loss or draw for the current player with perfect play
    - move:
        Number which is the optimal move that maximizes value for current player
        
Loads a training data set and a test data set for verification and checking for overfitting
        

"""

from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

#LOAD DATA AND EXTRACT KEYS, VALUES, AND MOVES
all_data = np.load( "data/training_set4.npz" )
hash_keys = all_data['hash_keys']
hash_vals = all_data['hash_vals']
hash_moves = all_data['hash_moves']

good_ind = np.nonzero(hash_keys)
hash_keys = hash_keys [ good_ind ]
hash_vals = hash_vals [ good_ind ]
hash_moves = hash_moves[good_ind ]


#change each entry in hash_keys from 64-bit int to 8 8-bit ints. 
#Then reshapes to Nx8. then removes the last column, so size is Nx7, because we only use 56 bits
reshaped_keys = np.reshape ( hash_keys.view(np.uint8), [len(hash_keys),8] )[:, :-1]

#change the 7 8-bit numbers to 56 values of '0' or '1'. array is now Nx56, ready for input into Neural Network
bit_keys = np.reshape ( np.unpackbits( reshaped_keys ), [ len(reshaped_keys), 56 ] )

#FORMAT THE OUTPUT: HASH VALS SO THAT THEY ARE EITHER +1, -1, OR ZERO
hash_vals [ np.where(hash_vals > 0) ] = 1
hash_vals [ np.where(hash_vals < 0) ] = -1
n_possible_moves = 7
moves_training = keras.utils.to_categorical(hash_moves, n_possible_moves)

#Split training and test data:
X_train, X_test, y_train, y_test = train_test_split( bit_keys, moves_training, test_size=0.10)

model = keras.Sequential()
model.add(keras.layers.Dense(56, input_shape = (56,),activation='tanh'))
model.add(keras.layers.Dense(30, activation='relu'))
model.add(keras.layers.Dense(20, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(n_possible_moves, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2)


predictions = model.predict ( X_test )
n_correct1 = 0 #correctly predicted moves
n_correct2 = 0 #almost correctly predicted moves (i.e. the second-best guess was correct)
n_wrong = 0

for i in range(len(predictions)):
    true_move = np.argmax ( y_test[i] )
    predicted_best_moves = [list(predictions[i]).index(x) for x in sorted(predictions[i], reverse=True)[:7]]
    if ( true_move == predicted_best_moves[0] ):
        n_correct1 += 1
    elif (true_move == predicted_best_moves[1]):
        n_correct2 += 1
    else:
        n_wrong += 1

print ( "%Correct1: ", n_correct1 / (n_correct1 + n_correct2 +n_wrong) )
print ( "%almost Correct1: ", (n_correct1 + n_correct2) / (n_correct1 + n_correct2 +n_wrong) )

#model.save("project_ANN2")

# test_key = 0
# test_key_bits = [1 if digit=='1' else 0 for digit in format(test_key,'056b')]
# test_input = np.array ( [test_key_bits] )
# test_move_prediction = model.predict ( test_input )

#Convolutional network below.
# model.add(keras.layers.Conv2D( 28,  kernel_size=(3, 3), strides=(1, 1),
#                  activation='sigmoid',input_shape=(8,7,1),data_format="channels_last" ,padding="same"))                 
# model.add( keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding="same") )
# model.add(keras.layers.Conv2D( 32,  kernel_size=(3, 3), activation='sigmoid' ,padding="same"))
# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),padding="same"))
