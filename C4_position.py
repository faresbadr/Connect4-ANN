"""

48 real positions, and 7 'imaginry positions' (Top row). Top row is used to tell us when a column is filled.
Bitboard encoded as:
* .  .  .  .  .  .  .
* 5 12 19 26 33 40 47
* 4 11 18 25 32 39 46
* 3 10 17 24 31 38 45
* 2  9 16 23 30 37 44
* 1  8 15 22 29 36 43
* 0  7 14 21 28 35 42

"""

import random
import numpy as np
from numba import jit

#constants
top_masks = [0x20 ,0x1000 ,0x80000, 0x4000000, 0x200000000, 0x10000000000 , 0x800000000000]
bottom_masks = [0x1, 0x80 , 0x4000,  0x200000,  0x10000000,   0x800000000,   0x40000000000]
height = 6
width = 7
all_bottom = (1<<0) | (1<<7) | (1<<14) | (1<<21) | (1<<28) | (1<<35) | (1<<42) #mask with bottom row of board = 1
board_mask = all_bottom * ((1 << height) - 1) #mask with '1' in every real board position. Thus, all positions '1' except the top imaginary row


""" returns a bitmask with winning free spots that make an alignment and are possible to play.
    Looks complicated, but if you draw a bitboard and follow the rules you'll see how they work out. uses JIT compiled numba for speed"""
@jit
def can_win_next( pos, mask ):
    #vertical
    r = ( pos << 1) & (pos << 2) & (pos << 3)
    
    #horizontal
    p = ( pos << ( height + 1)) & (pos << 2*(height + 1))
    r |= p & (pos << 3 * (height + 1))
    r |= p & (pos >> (height + 1))
    p = (pos >> (height + 1)) & (pos >> 2 * (height + 1))
    r |= p & (pos << (height + 1))
    r |= p & (pos >> 3 * (height + 1))
    
    #diagonal 1
    p = (pos << height) & (pos << 2 * height)
    r |= p & (pos << 3 * height)
    r |= p & (pos >> height)
    p = (pos >> height) & (pos >> 2 * height)
    r |= p & (pos << height)
    r |= p & (pos >> 3 * height)
    
    #diagonal 2
    p = (pos << (height + 2)) & (pos << 2 * (height + 2));
    r |= p & (pos << 3 * (height + 2));
    r |= p & (pos >> (height + 2));
    p = (pos >> (height + 2)) & (pos >> 2 * (height + 2));
    r |= p & (pos << (height + 2));
    r |= p & (pos >> 3 * (height + 2));
    winning_pos =  r & (board_mask ^ mask);
    possible = ( mask + all_bottom) & board_mask
    return winning_pos & possible 


""" returns true if 4 stones align in the given bitmask 'pos' """
def alignment(pos):
    m = pos & (pos >> (height+1)) #horizontal
    if(m & (m >> (2*(height+1)))):
        return True
    m = pos & (pos >> height) #diagonal 1
    if(m & (m >> (2*height))):
        return True
    m = pos & (pos >> (height+2)) #diagonal 2
    if(m & (m >> (2*(height+2)))):
        return True
    m = pos & (pos >> 1); #vertical
    if(m & (m >> 2)):
        return True
    return False

class C4_state:
    """
    Class which keeps track of the state of the game.
    This consists of a 'current_pos' and 'mask' bitmasks
    """
    def __init__(self, current=0, pos_mask=0, n_moves = 0 ):
        self.current_pos = current  #bitmap with '1' where there are current player stones
        self.mask = pos_mask        #bitmap with '1' anywhere there is a stone
        self.n_moves = n_moves
        
    #return number of '1's in a binary number. Don't use this function. it's too slow
    # def popcount(self,n):
    #     return bin(n).count("1")


    def reset(self):
        self.current_pos = 0
        self.mask = 0
        self.n_moves = 0
        
    """input: 0-based index of column to be played  """
    def play(self,move):
        self.current_pos ^= self.mask #switch current player with opponent
        self.mask |= self.mask + bottom_masks[move]
        self.n_moves += 1

    def play_string(self, col_string):
        for i in col_string:
            self.play( int(i) )

    def check_draw(self):
        """ Returns True if the game state is a draw """
        if self.n_moves == 42:
            return True
        else:
            return False
        
    
    def alignment(self,pos):
        """ returns true if 4 stones align in the given bitmask 'pos' """
        m = pos & (pos >> (height+1)) #horizontal
        if(m & (m >> (2*(height+1)))):
            return True
        m = pos & (pos >> height) #diagonal 1
        if(m & (m >> (2*height))):
            return True
        m = pos & (pos >> (height+2)) #diagonal 2
        if(m & (m >> (2*(height+2)))):
            return True
        m = pos & (pos >> 1); #vertical
        if(m & (m >> 2)):
            return True
        return False
        

    """return True if move will result in a loss next turn, False otherwise"""
    def is_losing_move(self,col):
        opponent_pos = C4_state( self.current_pos, self.mask, self.n_moves) # create a position from the current position, then play the move we're testing
        opponent_pos.play ( col)
        return bool ( can_win_next ( opponent_pos.current_pos, opponent_pos.mask ) ) #if opponent can win next, returns True
            
    def can_play(self,col):
        #Returns True if a column can be played, and false otherwise
        return (self.mask & top_masks[col])==0
    
    #""" returns a bitmask with all possible moves this turn"""
    def possible(self):
        return ( self.mask + all_bottom) & board_mask
    
    ''' starts with empty board. plays n random moves. if game ends before n moves, resets board and tries again '''
    def random_board(self, n_moves):
        self.current_pos = 0
        self.mask = 0
        possible_moves = [0,1,2,3,4,5,6]
        for turn in range( n_moves ):
            for move in range( len(possible_moves) ):
                rand_move = random.choice ( possible_moves )
                if self.can_play(rand_move):
                    self.play(rand_move)
                    break
                else:
                    possible_moves.remove(rand_move)
                
            last_pos = self.current_pos ^ self.mask    
            if self.alignment( last_pos): #check if the last move made an alignment
                self.random_board ( n_moves ) #if we made an alignment, then restart from the beginning
        return
                                        
    def display_board(self):
        if (self.n_moves%2):
            current_player = 2
        else:
            current_player = 1
            
        if (current_player==1):
            current_string = " X "
            opponent_string = " O "
        else: 
            current_string = " O "
            opponent_string = " X "
        printed_board = []
        for row_num in range( height ):
            one_row = []
            for col_num in range( width ):
                slot_value = 1<<row_num << (7*col_num)
                if (self.current_pos & slot_value):
                    one_row.append(current_string)
                elif ( (self.current_pos^self.mask) & slot_value):
                    one_row.append ( opponent_string )
                else:
                    one_row.append(" - ")
            printed_board.append(one_row)
        disp_board = np.array(printed_board)
        disp_board = np.flipud(disp_board)
        for row in disp_board:
            print (" ".join(map(str,row)))
        print (" 0   1   2   3   4   5   6 \n")


bench_pos = C4_state()
#bench_pos.play_string("222222")
#bench_pos.display_board()