
from math import sqrt
import copy
from sre_parse import GLOBAL_FLAGS

from numpy import append, insert
# import resource



import os           
import psutil

class Mem:
    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss
    
    # decorator function
    def profile(func):
        def wrapper(*args, **kwargs):
    
            mem_before = Mem.process_memory()
            result = func(*args, **kwargs)
            mem_after = Mem.process_memory()
            print("{}:consumed memory: {:,}".format(
                func.__name__,
                mem_before, mem_after, mem_after - mem_before))
    
            return result
        return wrapper
    
    
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

    def filter_value(self, state, row, column):
        values = range(1, self.size + 1)
        used = set()
        for i in range(0, self.size):
            used.add(state[row][i])
            used.add(state[i][column])

        square_row_start = int(row / self.square_size) * self.square_size
        square_column_start = int(column / self.square_size) * self.square_size
        for i in range(0, self.square_size):
            for j in range(0, self.square_size):
                used.add(state[square_row_start + i][square_column_start + j])
        return [number for number in values if number not in used]
    
    def find_empty_cell(self, state):
        for row in range(self.size):
            for column in range(self.size):
                if state[row][column] == 0:
                    return row, column
        return None, None

    def find_empty_cell_heuristic(self, state):     # find the cell which has at least choices
                                                    # otherwise, find_empty_cell: if meet condition cell = 0, it will return immediately
        result_row = None
        result_column = None
        min_valid_values = self.size
        
        for row in range(self.size):
            for column in range(self.size):
                if state[row][column] == 0: # traverse all cell has value 0 to find the cell has the least choices
                    valid_values = len(self.filter_value(state, row, column))
                    if valid_values < min_valid_values:
                        result_row = row
                        result_column = column
                        min_valid_values = valid_values
        return result_row, result_column

    def expand(self, state, history):
        row, column = self.find_empty_cell(state)
        if row == None or column == None: return None
        valid_value = self.filter_value(state, row, column)
        return_queue = [copy.deepcopy(state) for i in valid_value]
        for i in range(0, len(return_queue)):
            return_queue[i][row][column] = valid_value[i]
            return_queue[i] = (return_queue[i], history + [(row, column, valid_value[i], state)])

    
        return return_queue
    
    def expand_dfs_heuristic(self, state, history):
        row, column = self.find_empty_cell_heuristic(state)
        if row == None or column == None: return None
        valid_value = self.filter_value(state, row, column)
        return_stack = [copy.deepcopy(state) for i in valid_value]
        for i in range(0, len(return_stack)):
            return_stack[i][row][column] = valid_value[i]
            return_stack[i] = (return_stack[i], history + [(row, column, valid_value[i], state)])

        return return_stack
    
    # @Mem.profile 
    def bfs(self): 
        # file = open("temp.txt", "w")
        queue = [(copy.deepcopy(self.board), [])]
        state = None
        history = None
        while (len(queue) != 0):
            state = queue[0][0]
            history = queue[0][1]
            queue.pop(0)
            if self.check_goal(state): return state, history
            expand_queue = self.expand(state, history)
            if expand_queue == None: continue # Pass
            queue += expand_queue
                
        print("No solution!")
    
    def dfs_with_heuristic(self):
        # Heuristic in choose cell
        # Back tracking (DFS) in search
        stack = [(copy.deepcopy(self.board), [])]
        state = None
        history = None
        while (len(stack) != 0):
            stack_top = stack.pop()
            state = stack_top[0]
            history = stack_top[1]
            
            if self.check_goal(state): return state, history
            expand_stack = self.expand_dfs_heuristic(state, history)
            if expand_stack == None: continue
            stack = expand_stack + stack
            
        print("No solution!")
        
    # @Mem.profile
    def heuristic(self):
        return self.dfs_with_heuristic()

def online_init():
    level = -1
    while (level not in range(0, 6)):
        print("Input level of difficult: ")
        print(" BASIC - 0\n", "EASY - 1\n", "INTERMEDIATE - 2\n", "ADVANCE - 3\n", "EXTREAME - 4\n", "EVIL - 5")
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




# if __name__ == "__main__":
def main():
    board = Sudoku(online_init())
    print(board)
    import time
    
    start = time.time()
    result_bfs, history_bfs = board.bfs()
    end = time.time()
    print("BFS time: ", end - start)
    with open("sudoku_bfs_result.txt", "w") as output_file:
        output_file.write(Sudoku.result(result_bfs, history_bfs, board.size))
    
    start = time.time()
    result_heuristic, history_heuristic = board.heuristic()
    end = time.time()  
    print("Heuristic time: ", end - start) 
    with open("sudoku_heuristic_result.txt", "w") as output_file:
        output_file.write(Sudoku.result(result_heuristic, history_heuristic, board.size))
    
    # mem_heuristic = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print("Memory: ", mem_heuristic, " bytes")
    
    
    
      
main()


""" 
Normal DFS / BFS:
    State hiện tại gen ra các state kế (Tối ưu bằng cách chỉ gen state ở ô trống đầu tiên, không cần gen ở các ô khác)
    DFS / BFS các state đó

DFS áp dụng Heuristic:
    Từ state hiện tại chọn ra x state tốt nhất (tốt bằng nhau) bằng cách: 
        Lấy các state gen ra tại ô có ít lựa chọn nhất
    DFS các state đó

    Hàm đánh giá độ tốt: Số ô đã điền + Càng ít lựa chọn càng tốt
    ==> Gần giống Hill climbing vì chọn ra n state tốt nhất từ state hiện tại đi duyệt (Các state đó có độ tốt như nhau nên duyệt hết)
    ==> Đi DFS các state đó cũng là đi duyệt các state tốt hơn
    (Best first chọn ra state tốt nhất)
"""