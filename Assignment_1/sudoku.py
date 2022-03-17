
from math import sqrt
import numpy
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

    def print_board(self):
        for i in range(0, self.size):
            for j in range(0, self.size):
                print(self.board[i][j], end=" ")
                if ((j + 1) % int(sqrt(self.size)) == 0 and j < self.size - 1): print("|", end=" ")
            print()
            if ((i + 1) % int(sqrt(self.size)) == 0 and i < self.size - 1): print("---------------------")

def online_init():
    hard = -1
    while (hard not in range(0, 6)):
        print("Input level of difficult: ")
        print("BASIC - 0", "EASY - 1", "INTERMEDIATE - 2", "ADVANCE - 3", "EXTREAME - 4", "EVIL - 5")
        hard = int(input())
    import requests
    request = requests.get(f"https://www.puzzle-sudoku.com/?size={hard}")
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

board = Board(online_init(), "online")
board.print_board()
#board = Board(custom_init())