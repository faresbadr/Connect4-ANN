# -*- coding: utf-8 -*-
"""
Engine for Connect Four A.I
Functions:
        - Draw Boad
        - play_game

@author: Fares
"""

import solver #Finds optimal move by a minimax search
from tensorflow import keras #Chooses a move based on Neural Network recommendation
from C4_position import C4_state, can_win_next, alignment #a position is a unique board state
import numpy as np
import math
import sys

class engine:
    def __init__(self):   
        self.game_state = C4_state()
        #  LOAD THE GAME AI
        self.solver_AI = solver.solver()
        self.ANN_AI = keras.models.load_model("project_ANN2")
    
    def AI_move(self):
        # If there is an immediate winning move, take it
        winning_moves = can_win_next( self.game_state.current_pos, self.game_state.mask ) 
        if winning_moves:
            first_set_bit =  math.log2(winning_moves & -winning_moves)+1 #return index of first '1'
            move = math.floor( first_set_bit / 7) #winning column that should be played
            self.game_state.play (move)
            return
        
        # else use ANN to order moves and let it pick best move 
        key = self.game_state.current_pos + self.game_state.mask
        key_bits = [1 if digit=='1' else 0 for digit in format(key,'056b')]
        NN_input = np.array ( [key_bits] )
        NN_prediction = self.ANN_AI.predict ( NN_input )
        worst_to_best_move_order = np.argsort ( NN_prediction )[0]
        
        #Iterate through moves, best to worst. play the first legal one that doesn't immediately lose
        for move in reversed(worst_to_best_move_order):
            if (not self.game_state.can_play(move)) or self.game_state.is_losing_move(move):
                continue
            self.game_state.play (move)
            return
        
        #if no non-losing moves were found, pick any legal move
        for move in range(7):
            if( not self.game_state.can_play(move) ):
                continue
            self.game_state.play(move)
            return
    
    def human_move(self):
        # Keeps asking human to input a move until they input a valid move
        while True:
            move = input("Input move, integer from 0 to 6: (q to quit) \n")
            if move == 'q':
                sys.exit()
            try:
                move = int(move)
                if self.game_state.can_play(move):
                    self.game_state.play(move)
                    break
                else:
                    print("Move is not legal")
                    continue
            except:
                print ("input is not a valid number")
                continue
        return
    
    def play_game(self, human_first = True):
        """
        runs game until it is over or exited
        """
        self.game_state.reset()
        if human_first:
            self.human_move()
            self.game_state.display_board()
            
        while True:
            self.AI_move()
            self.game_state.display_board()
            
            #check if game is over by alignment or draw
            AI_pieces = self.game_state.current_pos ^ self.game_state.mask
            if alignment( AI_pieces ):
                print ("AI wins!")
                break
            if self.game_state.check_draw():
                print ("Draw!")
                return
            
            self.human_move()
            self.game_state.display_board()
            human_pieces = self.game_state.current_pos ^ self.game_state.mask
            if alignment( human_pieces ):
                print ("Human wins!")
                break
            if self.game_state.check_draw():
                print ("Draw!")
                return
            