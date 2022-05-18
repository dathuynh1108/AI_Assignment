"""
This is a modified version of the minimax algorithm for Tic Tac Toe from the following source:
Link: https://github.com/Cledersonbc/tic-tac-toe-minimax

An implementation of Minimax AI Algorithm in Tic Tac Toe, using Python.
This software is available under GPL license.
Author: Clederson Cruz
Year: 2017
License: GNU GENERAL PUBLIC LICENSE (GPL)
"""


from typing import Tuple, List, Optional
from math import inf
from random import choice
from copy import deepcopy
from boardclasses import TicTacToeBoard, LocalBoard, GlobalBoard
import numpy as np
from state import * 
COMP: int
HUMAN: int

import numpy as np


class UltimateTTT_Move:
    
    def __init__(self, index_local_board, x_coordinate, y_coordinate, value):
        self.index_local_board = index_local_board
        self.x = x_coordinate
        self.y = y_coordinate
        self.value = value
    

    def __repr__(self):
        return "local_board:{0}, (x:{1} y:{2}), value:{3}".format(
                self.index_local_board,
                self.x,
                self.y,
                self.value       
            )

        
class State:
    
    X = 1
    O = -1
    free_move = False
    
    def __init__(self, state = None): # init with 0 arg or 1 arg (state)
        if not state:
            self.global_cells = np.zeros(9)
            self.blocks = np.array([np.zeros((3, 3)) for x in range(9)])
            self.player_to_move: int = 1
            self.previous_move:  UltimateTTT_Move = None
        else:
            self.global_cells = np.copy(state.global_cells)
            self.blocks = np.copy(state.blocks)
            self.player_to_move: int = state.player_to_move
            self.previous_move: UltimateTTT_Move = state.previous_move
    
    
    def __repr__(self):        
        return '''player: {0} \n\nmove: {1} \n\nafter move:\n\n+ global cells: 
                \n\n{2}\n+ blocks:\n\n{3}
                \n#############################################\n'''.format(
                self.player_to_move * -1,
                self.previous_move,
                self.global_cells.reshape(3, 3),
                self.blocks
            )
    
    
    # game result on single board (local or global)
    def game_result(self, board):
        row_sum = np.sum(board, 1)
        col_sum = np.sum(board, 0)
        diag_sum_topleft = board.trace()
        diag_sum_topright = board[::-1].trace()
        
        player_one_wins = any(row_sum == 3) + any(col_sum == 3)
        player_one_wins += (diag_sum_topleft == 3) + (diag_sum_topright == 3)
        
        if player_one_wins:
            return self.X

        player_two_wins = any(row_sum == -3) + any(col_sum == -3)
        player_two_wins += (diag_sum_topleft == -3) + (diag_sum_topright == -3)

        if player_two_wins:
            return self.O
        
        if np.all(board != 0):
            return 0.
        
        # if not over
        return None
        
    @property    
    def game_over(self):
        return self.game_result(self.global_cells.reshape(3,3)) != None
    
    @property
    def get_valid_moves(self):
        if self.previous_move != None:
            index_local_board = self.previous_move.x * 3 + self.previous_move.y
        else: 
            temp_blocks = np.zeros((3, 3))
            indices = np.where(temp_blocks == 0)
            ret = []
            for i in range(9):
                ret += [UltimateTTT_Move(i, index[0], index[1], self.player_to_move)
                        for index in list(zip(indices[0], indices[1]))
                    ]
            return ret
            
        local_board = self.blocks[index_local_board]
        indices = np.where(local_board == 0)
        
        if(len(indices[0]) != 0):
            self.free_move = False
            return [UltimateTTT_Move(index_local_board, index[0], 
                                     index[1], self.player_to_move)
                    for index in list(zip(indices[0], indices[1]))
                ]
        # chosen board is full      
        self.free_move = True        
        ret = []
        for i in range(9):
            if not np.all(self.blocks[i] != 0):
                indices = np.where(self.blocks[i] == 0)
                ret += [UltimateTTT_Move(i, index[0], index[1], self.player_to_move)
                        for index in list(zip(indices[0], indices[1]))
                    ]
        return ret
        
    
    def is_valid_move(self, move: UltimateTTT_Move):            
        if move.value != self.player_to_move:
            return False
        
        if move.x not in range(3) or move.y not in range(3):
            return False
        
        if self.previous_move and (not self.free_move):
            if(move.index_local_board != (self.previous_move.x * 3 + self.previous_move.y)):
                return False

        board_to_move = self.blocks[move.index_local_board]
        return board_to_move[move.x, move.y] == 0 # check if board field not occupied yet
    
    
    def act_move(self, move: UltimateTTT_Move):
        if not self.is_valid_move(move):
            raise ValueError(
                "move {0} on local board is not valid".format(move)
            )
        local_board = self.blocks[move.index_local_board]
        local_board[move.x, move.y] = move.value
        
        self.player_to_move *= -1          
        self.previous_move = move
        
        if self.global_cells[move.index_local_board] == 0: # not 'X' or 'O'
            if self.game_result(local_board):
                self.global_cells[move.index_local_board] = move.value
                
        # print(self)
    
    @property
    def count_X(self):
        return len((np.where(self.global_cells == 1))[0])
    
    @property
    def count_O(self):
        return len((np.where(self.global_cells == -1))[0])
        
  
class State_2(State):
    def __init__(self, state = None):
        super().__init__(state)
    
    @property
    def get_valid_moves(self):
        is_occupied = False
        if self.previous_move != None:
            index_local_board = self.previous_move.x * 3 + self.previous_move.y
            if self.global_cells[index_local_board] != 0:
                is_occupied = True
        else:
            temp_blocks = np.zeros((3, 3))
            indices = np.where(temp_blocks == 0)
            ret = []
            for i in range(9):
                ret += [UltimateTTT_Move(i, index[0], index[1], self.player_to_move)
                        for index in list(zip(indices[0], indices[1]))
                    ]
            return ret
        
        local_board = self.blocks[index_local_board]
        indices = np.where(local_board == 0)
        
        if (len(indices[0]) != 0) and (not is_occupied):
            self.free_move = False
            return [UltimateTTT_Move(index_local_board, index[0], 
                                     index[1], self.player_to_move)
                    for index in list(zip(indices[0], indices[1]))
                ]
        # chosen block is full or occupied (1 or -1)      
        self.free_move = True        
        ret = []
        for i in range(9):
            if (self.global_cells[i] == 0) and (not np.all(self.blocks[i] != 0)):
                indices = np.where(self.blocks[i] == 0)
                ret += [UltimateTTT_Move(i, index[0], index[1], self.player_to_move)
                        for index in list(zip(indices[0], indices[1]))
                    ]
        return ret
import numpy as np
from numpy.lib.function_base import copy
from state import *

MAX = np.inf
MIN = -np.inf

def act_move_test(cur_state:State, move:UltimateTTT_Move):
    """Move without affect to real game"""
    cur_state.local_board = cur_state.blocks[move.index_local_board]
    cur_state.local_board[move.x, move.y] = move.value

    if cur_state.global_cells[move.index_local_board] == 0: # not 'X' or 'O'
        if cur_state.game_result(cur_state.local_board):
            cur_state.global_cells[move.index_local_board] = move.value

def evaluate_global(cur_state:State)->int:
    """Return evaluate score of a state"""
    score = cur_state.game_result(cur_state.global_cells.reshape(3, 3))
    if (score != None): return 10000 * score
    return heuristic_for_X(cur_state) + heuristic_for_O(cur_state)

def get_valid_moves_test(cur_state, player, previous_move):
        if previous_move != None:
            cur_state.index_local_board = previous_move.x * 3 + previous_move.y
        else: 
            temp_blocks = np.zeros((3, 3))
            indices = np.where(temp_blocks == 0)
            ret = []
            for i in range(9):
                ret += [UltimateTTT_Move(i, index[0], index[1], player)
                        for index in list(zip(indices[0], indices[1]))
                    ]
            return ret
            
        local_board = cur_state.blocks[cur_state.index_local_board]
        indices = np.where(local_board == 0)
        
        if(len(indices[0]) != 0):
            cur_state.free_move = False
            return [UltimateTTT_Move(cur_state.index_local_board, index[0], 
                                     index[1], player)
                    for index in list(zip(indices[0], indices[1]))
                ]
        # chosen board is full      
        cur_state.free_move = True        
        ret = []
        for i in range(9):
            if not np.all(cur_state.blocks[i] != 0):
                indices = np.where(cur_state.blocks[i] == 0)
                ret += [UltimateTTT_Move(i, index[0], index[1], player)
                        for index in list(zip(indices[0], indices[1]))
                    ]
        return ret

def act_move_test(cur_state:State, move:UltimateTTT_Move):
    """Move without affect to real game"""
    cur_state.local_board = cur_state.blocks[move.index_local_board]
    cur_state.local_board[move.x, move.y] = move.value

    if cur_state.global_cells[move.index_local_board] == 0: # not 'X' or 'O'
        if cur_state.game_result(cur_state.local_board):
            cur_state.global_cells[move.index_local_board] = move.value

def minimax(cur_state, depth, player, previous_move, alpha, beta)->int:
    """Return a score that state can lead to"""
    if depth == 3: return evaluate_global(cur_state)
    valid_moves = get_valid_moves_test(cur_state, player, previous_move)
    # print(valid_moves)
    if player == 1:
        bestVal = -np.inf
        for move in valid_moves:
            copy_state = State(cur_state)
            act_move_test(copy_state, move)
            pre_move = move
            value = minimax(copy_state, depth + 1, -1, pre_move, alpha, beta)
            bestVal = max(bestVal, value)
            alpha = max(alpha, bestVal)

            # Pruning
            if beta <= alpha:
                break

        return bestVal

    else:
        bestVal = np.inf
        for move in valid_moves:
            copy_state = State(cur_state)
            act_move_test(copy_state, move)
            pre_move = move
            value = minimax(copy_state, depth + 1, 1, pre_move, alpha, beta)
            bestVal = min(bestVal, value)
            beta = min(beta, bestVal)

            # Pruning
            if beta <= alpha:
                break

        return bestVal    


def check_half_winning(board, player):
    score = 0
    for i in range(3):
        if (board[i][0] == player and board[i][1] == player and board[i][2] == 0 or board[i][0] == player and board[i][1] == 0 and board[i][2] == player or board[i][0] == 0 and board[i][1] == player and board[i][2] == player):
            score += 4
    for j in range(3):
        if (board[0][j] == player and board[1][j] == player and board[2][j] == 0 or board[0][j] == player and board[1][j] == 0 and board[2][j] == player or board[0][j] == 0 and board[1][j] == player and board[2][j] == player):        
            score += 4
    if (board[0][0] == player and board[1][1] == player and board[2][2] == 0) or (board[0][0] == player and board[1][1] == 0 and board[2][2] == player) or (board[0][0] == 0 and board[1][1] == player and board[2][2] == player):
        score += 4
    if (board[0][2] == player and board[1][1] == player and board[2][0] == 0) or (board[0][2] == player and board[1][1] == 0 and board[2][0] == player) or (board[0][2] == 0 and board[1][1] == player and board[2][0] == player):          
        score += 4
    return score
       
def heuristic_for_O(cur_state:State):
    """evaluate the game is not over"""
    """O is minimal"""
    score = 0
    board = cur_state.global_cells.reshape(3, 3)
    for i in range(3):
        for j in range(3):
            if board[i][j] == -1:
                score -= 5
    if board[1][1] == -1: score -= 5
    if board[0][0] == -1: score -= 2
    if board[0][2] == -1: score -= 2
    if board[2][0] == -1: score -= 2
    if board[2][2] == -1: score -= 2
    for local_board in cur_state.blocks:
        if local_board[1][1] == -1: score -= 3
    center_small_board = cur_state.blocks[4]
    for i in range(3):
        for j in range(3):
            if center_small_board[i][j] == -1: score -= 3
    score -= check_half_winning(board, -1)
    for local_board in cur_state.blocks:
        score -= check_half_winning(local_board, -1)/2
    return score

def heuristic_for_X(cur_state:State):
    """evaluate the game is not over"""
    """X is maximal"""
    score = 0
    board = cur_state.global_cells.reshape(3, 3)
    for i in range(3):
        for j in range(3):
            if board[i][j] == 1:
                score += 5
    if board[1][1] == 1: score += 5
    if board[0][0] == 1: score += 2
    if board[0][2] == 1: score += 2
    if board[2][0] == 1: score += 2
    if board[2][2] == 1: score += 2
    for local_board in cur_state.blocks:
        if local_board[1][1] == 1: score += 3
    center_small_board = cur_state.blocks[4]
    for i in range(3):
        for j in range(3):
            if center_small_board[i][j] == 1: score += 3
    score += check_half_winning(board, 1)
    for local_board in cur_state.blocks:
        score += check_half_winning(local_board, 1)/2
    return score    


#X move
# tinh tong so quan X tren toan ban co
def count_X(state):
    res = 0
    for block in state.blocks:
        for i in range(3):
            for j in range(3):
                if block[i][j] == 1:
                    res += 1
    return res

# kiem tra xem mot local/global block co day chua
def is_full(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return False
    return True

# tim o can danh qua index cua block
def find_cell_by_idxboard(idx):
    if idx == 0:
        return [0, 0]
    if idx == 1:
        return [0, 1]
    if idx == 2:
        return [0, 2]
    if idx == 3:
        return [1, 0]
    if idx == 4:
        return [1, 1]
    if idx == 5:
        return [1, 2]
    if idx == 6:
        return [2, 0]
    if idx == 7:
        return [2, 1]
    if idx == 8:
        return [2, 2]

# tra ve index cua local block chua danh vao cell
def find_board_not_check_at_cell(state, cell):
    res = -1
    for i in range(9):
        res += 1
        if state.blocks[i][cell[0]][cell[1]] == 0:
            break
    return res

# tra ve idx cua block can don chinh
def idx_important_block(state):
    idx_block = -1
    for block in state.blocks:
        idx_block += 1
        if block[1][1] != 1:
            break
    return idx_block

#tim opposit block
def idx_opposite_block(cur_block):
    if cur_block == 0:
        return 8
    if cur_block == 1:
        return 7
    if cur_block == 2:
        return 6
    if cur_block == 3:
        return 5
    if cur_block == 5:
        return 3
    if cur_block == 6:
        return 2
    if cur_block == 7:
        return 1
    if cur_block == 8:
        return 0

# kiem tra xem neu khong the choi theo chien thuat cho quan X
def out_of_X_stategy(state, important_block, opposite_block, important_cell, 
                    opposite_cell, free_move, ixd_move_block):
    if (not free_move) and state.blocks[ixd_move_block][important_cell[0]][important_cell[1]] != 0 \
        and state.blocks[ixd_move_block][opposite_cell[0]][opposite_cell[1]] != 0:
        return True
    for i in range(9):
        if i == important_block or i == opposite_block or i == 4:
            continue
        for j in range(3):
            for k in range(3):
                if state.blocks[i][j][k] == -1:
                    return True
    return False
        
    

def X_move(cur_state):
    if cur_state.previous_move == None:
        return UltimateTTT_Move(4, 1, 1, 1)

    total_X = count_X(cur_state)
    idx_move_block = 3 * cur_state.previous_move.x + cur_state.previous_move.y
    if total_X <= 7:
        return UltimateTTT_Move(idx_move_block, 1, 1, 1)
    if total_X == 8:
        return UltimateTTT_Move(idx_move_block, cur_state.previous_move.x, cur_state.previous_move.y, 1)
    
    
    free_move = is_full(cur_state.blocks[idx_move_block])
    cur_state.free_move = free_move
    idx_important = idx_important_block(cur_state)
    idx_opposite = idx_opposite_block(idx_important)
    important_cell = find_cell_by_idxboard(idx_important)
    opposite_cell = find_cell_by_idxboard(idx_opposite)

      
    if out_of_X_stategy(cur_state, idx_important, idx_opposite, important_cell,
                         opposite_cell, free_move, idx_move_block):
        
        
        valid_moves = cur_state.get_valid_moves
        if len(valid_moves) != 0:
            return np.random.choice(valid_moves)
    
    
    if free_move:
           
        if cur_state.blocks[idx_opposite][important_cell[0]][important_cell[1]] == 0:
            
            idx_move_block = idx_opposite
            return UltimateTTT_Move(idx_move_block, important_cell[0], important_cell[1], 1)

        if cur_state.blocks[idx_opposite][opposite_cell[0]][opposite_cell[1]] == 0:
            
            idx_move_block = idx_opposite
            return UltimateTTT_Move(idx_move_block, opposite_cell[0], opposite_cell[1], 1)

        else:
            
            idx_move_block = find_board_not_check_at_cell(cur_state, important_cell)
            if idx_move_block != -1:
                
                return UltimateTTT_Move(idx_move_block, important_cell[0], important_cell[1], 1)
            
            
            idx_move_block = find_board_not_check_at_cell(cur_state, opposite_cell)
            return UltimateTTT_Move(idx_move_block, opposite_cell[0], opposite_cell[1], 1)
            
    else:
        
        if cur_state.blocks[idx_move_block][important_cell[0]][important_cell[1]] == 0:
            
            return UltimateTTT_Move(idx_move_block, important_cell[0], important_cell[1], 1)
        else:
            
            return UltimateTTT_Move(idx_move_block, opposite_cell[0], opposite_cell[1], 1)


def select_move(cur_state):
    if cur_state.player_to_move == 1:
        return X_move(cur_state)
    player = cur_state.player_to_move
    pre_move = cur_state.previous_move
    valid_moves = get_valid_moves_test(cur_state, -1, pre_move )
    best_move = valid_moves[0]
    for move in valid_moves:
        copy_state_1 = State(cur_state)
        act_move_test(copy_state_1, move)
        copy_state_2 = State(cur_state)
        act_move_test(copy_state_2, best_move)
        if (minimax(copy_state_1, 0, 1, pre_move, MIN, MAX) < minimax(copy_state_2, 0, 1, pre_move, MIN, MAX)):
            best_move = move
    print(best_move)
    return best_move
####################################################################### 
def heuristic(state: TicTacToeBoard, depth: int) -> int:
    """
    Heuristic evaluation of the current board state
    :param state: the current board state
    :param depth: the number of empty spaces left on the board. Preference is given for faster wins and slower losses.
    """
    if state.has_tic_tac_toe(COMP):
        score = depth + 1
    elif state.has_tic_tac_toe(HUMAN):
        score = -(depth + 1)
    else:  # draw/undetermined outcome
        score = 0
    return score


def get_empty_cells(state: TicTacToeBoard) -> List[Tuple[int, int]]:
    """
    Returns the coordinates of all the unclaimed spaces on the board
    :param state: the current board state
    :return: The coordinates of all the empty cells left on the board
    """
    cells = []
    for row_index, row in enumerate(state.board):
        for col_index, cell in enumerate(row):
            if cell == 0:
                cells.append((row_index, col_index))
    return cells



def bot_turn(global_board: GlobalBoard, bot: int):
    """
    Finds the next local board to play on. If undefined, it uses the minimax algorithm to decide which board to play on.
    Then uses the minimax algorithm again to decide where to play on that board.
    :param global_board: The entire global board of the game
    :param bot: is the bot player 1 or 2?
    :return: local_board and coordinates of the bot's next move
    """
    # Determine if the bot is player 1 or 2
    global COMP
    global HUMAN
    COMP = bot
    HUMAN = - bot
    
    local_board: Optional[LocalBoard] = None


    ################################
    state = State()
    state.global_cells = np.reshape(np.array(global_board.board), 9)
    # print("global_cells: ", state.global_cells)
    for i in range(9):
        state.blocks[i] = np.array(global_board.local_board_list[i].board)
        #state.blocks = np.array([x.board for x in global_board.local_board_list[i]])
    state.previous_move = global_board.previous_move
    state.player_to_move = bot

    ################################
    # If the next board is undetermined
    if all(lb.focus == lb.playable for lb in global_board.local_board_list): 
        state.free_move = True

    move = select_move(state)
    local_board_index = move.index_local_board
    row = move.x
    col = move.y
    global_board.previous_move = move
    return global_board.local_board_list[local_board_index], row, col

def random_turn(global_board: GlobalBoard, human: int):    
    local_board: Optional[LocalBoard] = None


    ################################
    state = State()
    state.global_cells = np.reshape(np.array(global_board.board), 9)
    # print("global_cells: ", state.global_cells)
    for i in range(9):
        state.blocks[i] = np.array(global_board.local_board_list[i].board)
        #state.blocks = np.array([x.board for x in global_board.local_board_list[i]])
    state.previous_move = global_board.previous_move
    state.player_to_move = human

    ################################
    # If the next board is undetermined
    if all(lb.focus == lb.playable for lb in global_board.local_board_list): 
        state.free_move = True
    move = select_move_random(state)
    local_board_index = move.index_local_board
    row = move.x
    col = move.y
    global_board.previous_move = move
    return global_board.local_board_list[local_board_index], row, col, local_board_index

def select_move_random(cur_state):
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        return np.random.choice(valid_moves)
    return None
