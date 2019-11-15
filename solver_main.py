# -*- coding: utf-8 -*-
"""
Uses the solver to solve randomly generated board states of Connect Four.
The solutions are saved to create a training set for an ANN

Uses the solver to produce a training set. Training set contains keys, values and moves
    - Key:
        Number that represents a unique board state
    - value:
        Number that represents if a board state is a win or loss or draw for the current player with perfect play
    - move:
        Number which is the optimal move that maximizes value for current player

"""
import solver
import numpy as np

n_games = 50 #number of games to be solved to produce the training set
n_random_moves = 16 #Number of moves played randomly before solver is invoked

test_solver = solver.solver()
test_solver.create_training_data( n_games, n_random_moves )

# change hash table structure to 3 arrays of keys, values and moves by removing zeros in the hash table
good_ind = np.nonzero(test_solver.hash_keys)
hash_keys = test_solver.hash_keys [ good_ind ]
hash_vals = test_solver.hash_vals [ good_ind ]
hash_moves = test_solver.hash_moves[good_ind ]

#np.savez( "test_set", hash_keys = hash_keys, hash_vals = hash_vals, hash_moves = hash_moves )