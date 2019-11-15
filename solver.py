"""
Solver uses negamax algorithm: 
Each player is trying to minimize the other's value (which at the same time maximizes their own)

Result of seach is a hash table with keys, values, and moves
given a board, returns its 'value' = number of turns to win/lose

"""

import numpy as np
from C4_position import C4_state, can_win_next, alignment


class solver:
    def __init__(self):
        self.width = 7 #board's dimensions
        self.height = 6
        
        #Masks used for computations
        self.top_masks = [0x20 ,0x1000 ,0x80000, 0x4000000, 0x200000000, 0x10000000000 , 0x800000000000]
        self.bottom_masks = [0x1, 0x80 , 0x4000,  0x200000,  0x10000000,   0x800000000,   0x40000000000]
        
        #hash table properties and initialization
        self.table_size = 15485867 #8388593
        self.hash_keys = np.zeros(self.table_size, dtype='uint64')
        self.hash_vals = np.zeros(self.table_size, dtype='int')
        self.hash_moves = np.zeros(self.table_size, dtype='uint8')
        
        #Benchmarks, from easiest to hardest
        self.bench0_string = "2021230311144455655432233441660"
        self.bench1_string = "20212303111444556554322334416"
        self.bench2_string = "202123031114445565543223344"
        self.bench3_string = "2021230311144455655432"
        self.bench4_string = "33333321544124"
        self.bench5_string = "333333215441"      
        self.bench6_string = "3333332154"
        
    ''' Calls negamaxa with iterative deepening and null window search: start with a min/max window and then narrow it down '''
    def iterative_eval(self, current_pos, mask, n_moves):
        min_val = -( 42 - n_moves ) // 2
        max_val =  ( 43 - n_moves ) // 2
        while (min_val < max_val):
            med_val = (min_val + max_val) // 2 #I don't really understand those 4 lines of code. roughly, we are picking 'med_val' between max_val and min_val
            if (med_val <= 0 and min_val//2 < med_val):
                med_val = min_val//2
            elif (med_val >= 0 and max_val//2 > med_val):
                med_val = max_val//2
            
            r = self.negamax(current_pos, mask, n_moves, med_val, med_val+1)
            if(r <= med_val):
                max_val = r
            else:
                min_val = r
            
        return min_val
    
    def solve(self, current_pos, mask, n_moves):
        scores_array = []
        for move in range(self.width):
            if mask & self.top_masks[move] != 0: #if can't play (move), skip
                scores_array.append('X')
                continue

            #play the move, then return negative of opponent's score
            new_pos, new_mask = self.play(current_pos, mask, move)
            new_n = n_moves + 1
            
            if alignment ( new_pos ^ new_mask ): #if move makes an alignment, then we know its score
                score = ( 43 - n_moves ) // 2
            else:
                score = - self.iterative_eval ( new_pos, new_mask, new_n)
                #score = - negamax( new_pos, new_mask, new_n, -beta, -alpha )
            scores_array.append(score)
        return scores_array
        
    def benchmark(self, bench_string):
        test_board = C4_state()
        test_board.play_string( bench_string )
        current_pos = test_board.current_pos
        mask = test_board.mask
        n_moves= test_board.n_moves  
        return self.solve( current_pos, mask, n_moves)   

    ''' for n games, creates a random board with specified number of random moves
        then calls negamax on each of those games, which fills the hash tables
        it may be a better idea to set the terminating condition as number of values in training array
    '''
    def create_training_data(self, n_games, n_random_moves):
        for game_counter in range(n_games):
            rand_pos = C4_state()  #create random board
            rand_pos.random_board(n_random_moves)
            self.iterative_eval(rand_pos.current_pos, rand_pos.mask, rand_pos.n_moves)  # plays game, by solving the position for its score
            print ("played one game")

    def play(self,pos, mask, move):
        pos ^= mask
        mask |= (mask + self.bottom_masks[move] )
        return pos,mask

    """
    Alpha represents the best score guaranteed for the current player
    Beta represents the worst score that the opponent can force current player into
    
    If alpha exceeds beta, search terminates because the opponent can force the game to a score of beta
    """
    def negamax(self,current_pos, mask, n_moves, alpha, beta):
        if( n_moves == 42): # check for draw. if so, return 0
            return 0
            
        if can_win_next(current_pos, mask): #if we end the game here, then we know the score. return score.
            return ( ( 43 - n_moves ) ) // 2 #integer division by 2 
    
    
        # GET UPPER BOUND OF SCORE. USE TO UPDATE BETA
        max_score = (41 - n_moves)//2  #Get upper bound of score
        highscore = -1 * max_score
        best_move = 0
        
        key = current_pos + mask #this operation produces a unique key for each board
        index = key%self.table_size
        looked_up_key = self.hash_keys[index]
        if (looked_up_key == key):
            max_score = self.hash_vals[index]
            
        if (beta > max_score):
            beta = max_score # no need to keep beta above maximum
            if alpha>=beta:
                return beta #terminate if [alpha;beta] is empty
           
        for move in [3,2,4,1,5,0,6]:
            if mask & self.top_masks[move] != 0: #if can't play (move), skip
                continue
            new_pos, new_mask = self.play(current_pos, mask, move)
            new_n = n_moves + 1
            score = - self.negamax( new_pos, new_mask, new_n, -beta, -alpha ) 
            
            if score > highscore:
                best_move = move
                highscore = score
            
            if score>=beta:
                return score  #PRUNE WHEN WE FIND A MOVE BETTER THAN SCORE THAT OPPONENT CAN FORCE US INTO
                
            if score>alpha:
                alpha = score  #ONLY TRACK SCORES BETTER THAN THE BEST SO FAR
            
        self.hash_keys[index] = key #once we evaluated a position, we keep its score in the transposition table 
        self.hash_vals[index] = alpha
        self.hash_moves[index] = best_move
        return alpha

