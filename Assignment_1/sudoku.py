
from math import sqrt
import copy

class Sudoku:
    def __init__(self, board):
        if isinstance(board, str):
            self.board = []
            for c in board:
                if c == "_": pass
                elif c >= "1" and c <= "9":
                    self.board.append(int(c))
                else:
                    self.board += [0 for i in range((ord(c) - ord("a") + 1))]
            self.size = int(sqrt(len(self.board)))
            self.board = [self.board[i : i + self.size] for i in range(0, len(self.board), self.size)]
        else:
            self.board = board
            self.size = len(board)
            for i in range(0, len(self.board)):
                board[i] = list(map(lambda c: 0 if c == "." else int(c), board[i]))
        self.square_size = int(sqrt(self.size))

    def __str__ (self):
        return Sudoku.board_to_string(self.board, self.size)
    
    @staticmethod    
    def board_to_string(board, size):
        square_size = int(sqrt(size))
        result = ""
        for i in range(0, size):
            for j in range(0, size):
                result += str(board[i][j]) + " " if board[i][j] != 0 else ". "
                if ((j + 1) % square_size == 0 and j < size - 1): result += "| "
            result+= "\n"
            if ((i + 1) % square_size == 0 and i < size - 1): result += "---------------------\n"
        return result
    
    @staticmethod
    def result(state, history, size):
        result = ""
        for history_state in history:
            result += Sudoku.board_to_string(history_state[3], size) + "\n"+ f"Row: {history_state[0]}, Column: {history_state[1]}, Value: {history_state[2]}\n"
        result += Sudoku.board_to_string(state, size)
        return result
    
    def check_goal(self, state):
        expect = sum(range(1, self.size + 1))
        # Check row and column:
        for i in range(0, self.size):
            if 0 in state[i]: return False
            if sum(state[i]) != expect: return False
            sum_column = 0
            for j in range(0, self.size):
                if state[j][i] == 0: return False
                sum_column += state[j][i]
            if sum_column != expect: return False

        # Check all sub square
            for row in range (0, self.size, self.square_size):
                for column in range(0, self.size, self.square_size):
                    square_sum = 0
                    for sub_row in range(0, self.square_size):
                        for sub_column in range(0, self.square_size):
                            square_sum += state[sub_row][sub_column]
                    if square_sum != expect: return False
        return True

    def find_empty_cell(self, state):
        for row in range(self.size):
            for col in range(self.size):
                if state[row][col] == 0:
                    return row, col
        return None, None

    def filter_row(self, state, row, column):
        pass

    def filter_column(self, state, row, column):
        pass

    def filter_subsquare(self, state, row, column):
        pass
    
    def expand(self, state, history):
        values = range(1, self.size + 1)
        row, column = self.find_empty_cell(state)
        
        if row == None or column == None: return None
        used = set()
        for i in range(0, self.size):
            used.add(state[row][i])
            used.add(state[i][column])
        
        square_row_start = int(row / self.square_size) * self.square_size
        square_column_start = int(column / self.square_size) * self.square_size
        for i in range(0, self.square_size):
            for j in range(0, self.square_size):
                used.add(state[square_row_start + i][square_column_start + j])
        valid_value = [number for number in values if number not in used]
        return_queue = [copy.deepcopy(state) for i in valid_value]
        for i in range(0, len(return_queue)):
            return_queue[i][row][column] = valid_value[i]
            return_queue[i] = (return_queue[i], history + [(row, column, valid_value[i], state)])
        return return_queue
    
    def bfs(self): 
        queue = [(copy.deepcopy(self.board), [])]
        state = None
        history = None
        while (len(queue) != 0):
            state = queue[0][0]
            history = queue[0][1]
            queue.pop(0)
            if self.check_goal(state): return state, history
            expand_queue = self.expand(state, history)
            if expand_queue == None: 
                print("No valid move")
                return
            queue += expand_queue

def online_init():
    level = -1
    while (level not in range(0, 6)):
        print("Input level of difficult: ")
        print("BASIC - 0", "EASY - 1", "INTERMEDIATE - 2", "ADVANCE - 3", "EXTREAME - 4", "EVIL - 5")
        level = int(input())
    import requests
    # Take task string from web page
    request = requests.get(f"https://www.puzzle-sudoku.com/?size={level}")
    task = request.text[request.text.find("var task ="):]
    task = task[task.find("'") + 1: task.find(";") - 1]
    return task

def custom_init():
    with open("sudoku_init.txt", encoding="utf-8") as init_file:
        board = []
        for line in init_file:
            line = line.strip()
            board.append(line.split(" "))
    return board

if __name__ == "__main__":
    board = Sudoku(custom_init())
    print(board)
    import time
    start = time.time()
    result, history = board.bfs()
    end = time.time()
    print("Search time: ", end - start)
    with open("sudoku_result.txt", "w") as output_file:
        output_file.write(Sudoku.result(result, history, board.size))

