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
from numpy.lib.function_base import copy


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

def minimax(cur_state, depth, player, previous_move, alpha, beta, max_depth)->int:
    """Return a score that state can lead to"""
    if depth == max_depth: return evaluate_global(cur_state)
    valid_moves = get_valid_moves_test(cur_state, player, previous_move)
    # print(valid_moves)
    if player == 1:
        bestVal = -np.inf
        for move in valid_moves:
            copy_state = State(cur_state)
            act_move_test(copy_state, move)
            pre_move = move
            value = minimax(copy_state, depth + 1, -1, pre_move, alpha, beta, max_depth)
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
            value = minimax(copy_state, depth + 1, 1, pre_move, alpha, beta, max_depth)
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


def select_move(cur_state, level):
    if level == "Beginner": max_depth = 1
    elif level == "Normal": max_depth = 2
    else: max_depth = 3
    
    if cur_state.player_to_move == 1: 
        if level == "Hard": return X_move(cur_state)
        if cur_state.previous_move == None: return UltimateTTT_Move(4, 1, 1, 1)        
        valid_moves = get_valid_moves_test(cur_state, 1, cur_state.previous_move)
        best_move = valid_moves[0]
    
        copy_state = State(cur_state)
        act_move_test(copy_state, best_move)
        best_move_minimax =  minimax(copy_state, 0, 1, cur_state.previous_move, MIN, MAX, max_depth)
    
        for move in valid_moves:
            copy_state_1 = State(cur_state)
            act_move_test(copy_state_1, move)
            cur_move_minimax = minimax(copy_state_1, 0, 1, cur_state.previous_move, MIN, MAX, max_depth)
            if (cur_move_minimax > best_move_minimax):
                best_move = move
                best_move_minimax = cur_move_minimax
        return best_move

    valid_moves = get_valid_moves_test(cur_state, -1, cur_state.previous_move )
    best_move = valid_moves[0]
    
    copy_state = State(cur_state)
    act_move_test(copy_state, best_move)
    best_move_minimax =  minimax(copy_state, 0, 1, cur_state.previous_move, MIN, MAX, max_depth)
    
    for move in valid_moves:
        copy_state_1 = State(cur_state)
        act_move_test(copy_state_1, move)
        cur_move_minimax = minimax(copy_state_1, 0, 1, cur_state.previous_move, MIN, MAX, max_depth)
        if (cur_move_minimax < best_move_minimax):
            best_move = move
            best_move_minimax = cur_move_minimax
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



def bot_turn(global_board: GlobalBoard, bot: int, level):
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
    move = select_move(state, level)
    local_board_index = move.index_local_board
    row = move.x
    col = move.y
    global_board.previous_move = move
    return global_board.local_board_list[local_board_index], row, col

def random_turn(global_board: GlobalBoard, human: int):    
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
    # if all(lb.focus == lb.playable for lb in global_board.local_board_list): 
    #     state.free_move = True
    
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
