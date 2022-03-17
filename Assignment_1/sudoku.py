
from math import sqrt
import copy

class Board:
    def __init__(self, board, type = "online"):
        if type == "online":
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
        self.square_size = int(sqrt(self.size))
        self.history = []

    def __str__ (self):
        return Board.board_to_string(self.board, self.size)
    
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

    def print_result(self):
        for history in self.history:
            print(Board.board_to_string(history[3], self.size))
            print(f"Row: {history[0]}, Column: {history[1]}, Value: {history[2]}")
        print(self.__str__())

    def check_goal(self):
        expect = sum(range(1, self.size + 1))
        # Check row and column:
        for i in range(0, self.size):
            if 0 in self.board[i]: return False
            if sum(self.board[i]) != expect: return False
            sum_column = 0
            for j in range(0, self.size):
                if self.board[j][i] == 0: return False
                sum_column += self.board[j][i]
            if sum_column != expect: return False

        # Check all sub square
            for row in range (0, self.size, self.square_size):
                for column in range(0, self.size, self.square_size):
                    square_sum = 0
                    for sub_row in range(0, self.square_size):
                        for sub_column in range(0, self.square_size):
                            square_sum += self.board[sub_row][sub_column]
                    if square_sum != expect: return False
        return True

    def find_empty_cell(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == 0:
                    return row, col
        return None, None

    def expand(self):
        values = range(1, self.size + 1)
        row, column = self.find_empty_cell()
        
        if row == None or column == None: return None
        used = set()
        for i in range(0, self.size):
            used.add(self.board[row][i])
            used.add(self.board[i][column])
        
        square_row_start = int(row / self.square_size) * self.square_size
        square_column_start = int(column / self.square_size) * self.square_size
        for i in range(0, self.square_size):
            for j in range(0, self.square_size):
                used.add(self.board[square_row_start + i][square_column_start + j])
        valid_value = [number for number in values if number not in used]
        return_queue = [copy.deepcopy(self.board) for i in valid_value]
        for i in range(0, len(return_queue)):
            return_queue[i][row][column] = valid_value[i]
            return_queue[i] = (return_queue[i], self.history + [(row, column, valid_value[i], self.board)])
        return return_queue
    
    def bfs(self): 
        queue = [(copy.deepcopy(self.board), [])]
        while (len(queue) != 0):
            self.board = queue[0][0]
            self.history = queue[0][1]
            queue.pop(0)
            if self.check_goal(): return self.board, self.history
            new_queue = self.expand()
            if new_queue == None: 
                print("No valid move")
                return
            queue += new_queue

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
        size = int(init_file.readline())
        board = []
        while True:
            current_line = init_file.readline()
            if not current_line: break
            board.append(current_line.split(" "))
    return board


if __name__ == "__main__":
    board = Board(online_init(), "online")
    board.bfs()
    board.print_result()

