from inspect import formatargspec
from math import sqrt
from copy import deepcopy
from multiprocessing.spawn import old_main_modules
import time

from numpy import number



count1 = 0
count2 = 0
count3 = 0
count4 = 0

result = ""
block = ['0', '1', '2', '3', '4', 'B']
class Light_Up:

    def __init__(self, board):
        if isinstance(board, str):         
            self.board = []
            for c in board:
                if c >= "0" and c <= "4": # Black block with number
                    self.board.append(c)
                elif c == "B": # Black block without number
                    self.board.append('B')
                else:
                    self.board += ['-' for i in range((ord(c) - ord("a") + 1))]
            self.size = int(sqrt(len(self.board)))
            self.board = [self.board[i : i + self.size] for i in range(0, len(self.board), self.size)]
            self.black_cell_with_number = []
            for i in range(len(self.board)):
                for j in range(len(self.board[i])):
                    if self.board[i][j] in block and self.board[i][j] != 'B' and self.board[i][j] != '0':
                        self.black_cell_with_number += [[i, j]]
        else:
            self.board = board
            self.size = len(self.board)
            self.black_cell_with_number = []
            for i in range(len(self.board)):
                for j in range(len(self.board[i])):
                    if self.board[i][j] in block and self.board[i][j] != 'B' and self.board[i][j] != '0':
                        self.black_cell_with_number += [[i, j]]
        
    def __str__ (self):
        return Light_Up.board_to_string(self.board, self.size)
    
    @staticmethod    
    def board_to_string(board, size):
        result = ""
        for i in range(0, size):
            for j in range(0, size):
                result += (board[i][j] + " ")
            result+= "\n"
        return result
    
class Solution:
    board = []
    child = []
    black_cell_with_number = []
    def __init__(self, board, child, black_cell_with_number):
        self.board = board
        self.size = len(self.board)
        self.child = child
        self.black_cell_with_number = black_cell_with_number
    
    def check_goal(self): 
        for i in self.board:
            for j in i:
                if j == "-":
                    return False
        if self.check_satisfy_all_black_cells() == False:
            return False
        return True

    def check_satisfy_all_black_cells(self):
        for i in range (0, self.size):
            for j in range (0, self.size):
                if (self.board[i][j] in block) and (self.board[i][j] != 'B'):
                    if self.check_satisfy_one_black_cell(i, j, int(self.board[i][j])) != 0:
                        return False    # the number of bulbs != the number in the black cell
        return True

    def check_satisfy_one_black_cell(self, row, col, number_in_black_cell): # check 4 corners of the cell to find bulbs
        bulbs_list = []
        if row > 0:
            bulbs_list.append(self.board[row - 1][col])
        if row < self.size - 1:
            bulbs_list.append(self.board[row + 1][col])
        if col > 0:
            bulbs_list.append(self.board[row][col - 1])
        if col < self.size - 1:
            bulbs_list.append(self.board[row][col + 1])
        return abs(number_in_black_cell - bulbs_list.count("x"))   # cuz bulbs can be more or less than the number in black cell ---> abs

    # kiểm tra 1 ô đen đc đánh 0,1,2,3 có nhiêu bóng đèn xung quanh nó
    def count_remaning_bulbs(self, row, col, number_in_black_cell): # check the black cell is having how many bulbs around 4 corners
        bulbs_list = []
        if row > 0:
            bulbs_list.append(self.board[row - 1][col])
        if row < self.size - 1:
            bulbs_list.append(self.board[row + 1][col])
        if col > 0:
            bulbs_list.append(self.board[row][col - 1])
        if col < self.size - 1:
            bulbs_list.append(self.board[row][col + 1])
        return number_in_black_cell - bulbs_list.count("x")    # check the number of bulbs that can be filled more, if < 0 --> false
    
    # khi đặt 1 bóng đèn vào ô (i,j) số bóng đèn xung quanh ở các ô đen chung cạnh ô (i,j) sẽ thay đổi
    # ta cần phải kiểm tra số bóng đèn xung quanh của 4 ô đen chung cạnh ô (i,j)
    def check_not_conflict_other_black_cells(self, row, col):
        if row > 0 and (self.board[row - 1][col] in block) and (self.board[row - 1][col] != 'B'):
            if self.count_remaning_bulbs(row - 1, col, int(self.board[row - 1][col])) < 0:
                return False
        if row < self.size - 1 and (self.board[row + 1][col] in block) and (self.board[row + 1][col] != 'B'):
            if self.count_remaning_bulbs(row + 1, col, int(self.board[row + 1][col])) < 0:
                return False
        if col > 0 and (self.board[row][col - 1] in block) and (self.board[row][col - 1] != 'B'):
            if self.count_remaning_bulbs(row, col - 1, int(self.board[row][col - 1])) < 0:
                return False
        if col < self.size - 1 and (self.board[row][col + 1] in block) and (self.board[row][col + 1] != 'B'):
            if self.count_remaning_bulbs(row, col + 1, int(self.board[row][col + 1])) < 0:
                return False
        return True
    
    def light_up_row_and_col(self, row, col, return_board):
        walker = row - 1
        while walker >= 0 and self.board[walker][col] not in block:
            return_board[walker][col] = "L"
            walker = walker - 1
        
        # light up the column from the cell 'x' (placed bulb) to bottom
        walker = row + 1
        while walker <= self.size - 1 and self.board[walker][col] not in block:
            return_board[walker][col] = "L"
            walker = walker + 1
            
        # light up the row from the cell 'x' (placed bulb) to left
        walker = col - 1
        while walker >= 0 and self.board[row][walker] not in block:
            return_board[row][walker] = "L"
            walker = walker - 1
        
        # light up the row from the cell 'x' (placed bulb) to right
        walker = col + 1
        while walker <= self.size - 1 and self.board[row][walker] not in block:
            return_board[row][walker] = "L"
            walker = walker + 1
            
        return return_board    # return new board after placed bulb and light up

    def put_bulb_and_light_up(self, row, col): 
        return_board = deepcopy(self.board)
        return_board[row][col] = "x"
        return self.light_up_row_and_col(row, col, return_board)
    
    # xóa từ danh sách black_cell_with_number ô đc đánh 1,2,3 mà đã đủ bóng đèn xung quanh nó
    def delete_satisfied_black_cells(self):
        i = 0
        while i < len(self.black_cell_with_number):     # check all black cell with number 1,2,3
            row = self.black_cell_with_number[i][0]   # row
            col = self.black_cell_with_number[i][1]   # col
            if self.count_remaning_bulbs(row , col, int(self.board[row][col])) == 0:    # completely filled all bulbs that satisfy condition number of the black cell
                self.black_cell_with_number.pop(i)
            i = i + 1

    def expand(self):
        return_child_list = []
        black_cell_with_number_copy = deepcopy(self.black_cell_with_number)
        m = deepcopy(self)
        
        # ta sẽ ưu tiên thăm các ô đen đc đánh số 1,2,3 mà ô đc đánh số i có ÍT HƠN i bóng đèn xung quanh
        # a chính là list những ô như vậy
        # Nếu a rỗng, khi này mọi ô đen đc đánh 1,2,3 đều đã đủ các bóng đèn xung quanh nó
        # Khi này chọn ô trắng đầu tiên chưa đc chiếu sáng, ta visit tất cả các ô mà chiếu sáng ô trắng này
        # như vậy trường hợp xấu nhất ta tạo 13 child (là những ô chung hàng và cột) 
        if black_cell_with_number_copy == []:
            found = False
            for i in range (0, self.size):
                for j in range (0, self.size):
                    if self.board[i][j] == "-":
                        new_board = m.put_bulb_and_light_up(i, j)
                        new_solution = Solution(new_board, self.child, black_cell_with_number_copy)
                        check_new_solution = new_solution.check_not_conflict_other_black_cells(i,j)
                        if check_new_solution == True:
                            return_child_list.append(new_solution)
                            
                        walker = i - 1
                        while walker >= 0 and self.board[walker][j] not in block:
                            if self.board[walker][j]=="-":
                                new_board = m.put_bulb_and_light_up(walker,j)
                                new_solution = Solution(new_board,self.child, black_cell_with_number_copy)
                                check_new_solution = new_solution.check_not_conflict_other_black_cells(walker,j)
                                if check_new_solution == True:
                                    return_child_list.append(new_solution)
                            walker = walker - 1
                            
                        walker = i + 1
                        while walker <= self.size - 1 and self.board[walker][j] not in block:
                            if self.board[walker][j]=="-":
                                new_board = m.put_bulb_and_light_up(walker,j)
                                new_solution = Solution(new_board,self.child, black_cell_with_number_copy)
                                check_new_solution = new_solution.check_not_conflict_other_black_cells(walker,j)
                                if check_new_solution == True:
                                    return_child_list.append(new_solution)
                            walker = walker + 1
                            
                        walker = j - 1
                        while walker >= 0 and self.board[i][walker] not in block:
                            if self.board[i][walker] == "-":
                                new_board = m.put_bulb_and_light_up(i,walker)
                                new_solution = Solution(new_board,self.child, black_cell_with_number_copy)
                                check_new_solution = new_solution.check_not_conflict_other_black_cells(i,walker)
                                if check_new_solution == True:
                                    return_child_list.append(new_solution)
                            walker = walker - 1
                            
                        walker = j + 1
                        while walker <= self.size - 1 and self.board[i][walker] not in block:
                            if self.board[i][walker] == "-":
                                new_board = m.put_bulb_and_light_up(i,walker)
                                new_solution = Solution(new_board,self.child, black_cell_with_number_copy)
                                check_new_solution = new_solution.check_not_conflict_other_black_cells(i,walker)
                                if check_new_solution == True:
                                    return_child_list.append(new_solution)  
                            walker = walker + 1
                            
                        break
                if found == True:
                    break
        # nếu a khác rỗng, khi này sẽ có 1 ô đen chưa đủ bóng đèn xung quanh nó, khi này ta
        # sẽ visit 4 ô xung quanh ô đen, vậy trường hợp xấu nhất ta tạo ra 4 child
        else:
            # example: a = [ [1, 4], [2, 1] ...]; 1 and 4 is index of row and column of black cell
            row_and_col = black_cell_with_number_copy[0]  # get the first pair [row, col]
            row = row_and_col[0] # row
            col = row_and_col[1] # col
            
            # global count3
            # if count3 != 100:
            #     count3 +=1
            #     print("row, col: ", row, col)
            
            # print("row, col: ", row, col)
            
            if row - 1 >= 0 and self.board[row - 1][col] == "-":
                new_board = m.put_bulb_and_light_up(row - 1, col)  # m is root (board, child, black_cell_with_number) ---> return a board (b)
                new_solution = Solution(new_board, self.child, black_cell_with_number_copy)
                check_new_solution = new_solution.check_not_conflict_other_black_cells(row - 1, col)            
                if check_new_solution == True:
                    new_solution.delete_satisfied_black_cells()
                    return_child_list.append(new_solution)
                    
            if row + 1 <= self.size - 1 and self.board[row + 1][col]=="-":
                new_board = m.put_bulb_and_light_up(row + 1,col)
                new_solution = Solution(new_board,self.child, black_cell_with_number_copy)
                check_new_solution = new_solution.check_not_conflict_other_black_cells(row + 1,col)            
                if check_new_solution == True:
                    new_solution.delete_satisfied_black_cells()
                    return_child_list.append(new_solution)
                    
            if col - 1 >= 0 and self.board[row][col - 1]=="-":
                new_board = m.put_bulb_and_light_up(row, col - 1)
                new_solution = Solution(new_board, self.child, black_cell_with_number_copy)
                check_new_solution = new_solution.check_not_conflict_other_black_cells(row,col - 1)            
                if check_new_solution == True:
                    new_solution.delete_satisfied_black_cells()
                    return_child_list.append(new_solution)  
                    
            if col + 1 <= self.size - 1 and self.board[row][col + 1]=="-":
                new_board = m.put_bulb_and_light_up(row, col + 1)
                new_solution = Solution(new_board, self.child, black_cell_with_number_copy)
                check_new_solution = new_solution.check_not_conflict_other_black_cells(row, col + 1)            
                if check_new_solution == True:
                    new_solution.delete_satisfied_black_cells()
                    return_child_list.append(new_solution) 
                     
        return return_child_list

    def print_board(self):
        for i in range(0,self.size):
            for j in range(0,self.size):
                print(self.board[i][j],end="")
                print(" ",end="")
            print()

    


def breadth_first_search(root):
    visited = []
    queue = []
    queue.append(root)
    visited.append(root.board)        # board: [ [], [] ...] ---> append: [ [ [], [] ...] ]
    
    while queue and (queue[0].check_goal() == False):
        state = queue.pop(0)
        state.child = state.expand()
        
        # a.print_board()
        # print()
        # print("---------------------------------")
        # print()

        for ele in state.child:
            if ele.board not in visited:
                queue.append(ele)
                visited.append(ele.board)

    queue[0].print_board()
    
    
def heuristic(solution):
    res = 0
    for i in range(0, solution.size):
        for j in range(0, solution.size):
            if solution.board[i][j]=='-':
                res = res + 1
            elif solution.board[i][j] in block and solution.board[i][j] != "B":
                res = res + solution.check_satisfy_one_black_cell(i, j, int(solution.board[i][j]))
                
    return res
    

#cũng giống breadth first nhưng sort queue sau mỗi lần thêm
def best_first_search(root):
    visited = []
    queue = []
    queue.append(root)
    visited.append(root.board)
    
    while queue and (queue[0].check_goal()==False):
        # a is root (board, child, black_cell_with_number)
        state = queue.pop(0)
        state.child = state.expand()
        state.child.sort(key=heuristic)


        # print()
        # print("---------------------------------")
        # print()
        for ele in state.child:
            if ele.board not in visited:
                queue.append(ele)
                visited.append(ele.board)
        queue.sort(key=heuristic)
                   
    queue[0].print_board()

    global result
    for i in range(len(queue[0].board)):
        for j in range(len(queue[0].board[i])):
            result += (queue[0].board[i][j] + " ")
        result +='\n'
        

def online_init():
    level = -1
    while (level not in range(0, 12)):
        print("Input your option:")
        print(" 0:   7x7 Light Up Easy")
        print(" 1:   7x7 Light Up Normal")
        print(" 2:   7x7 Light Up Hard")
        print(" 3:   10x10 Light Up Easy")
        print(" 4:   10x10 Light Up Normal")
        print(" 5:   10x10 Light Up Hard")
        print(" 6:   14x14 Light Up Easy")
        print(" 7:   14x14 Light Up Normal")
        print(" 8:   14x14 Light Up Hard")
        print(" 9:   25x25 Light Up Easy")
        print(" 10:  25x25 Light Up Normal")
        print(" 11:  25x25 Light Up Hard")
        level = int(input())
    import requests
    # Take task string from web page
    request = requests.get(f"https://www.puzzle-light-up.com/?size={level}")
    task = request.text[request.text.find("var task ="):]
    task = task[task.find("'") + 1: task.find(";") - 1]
    return task

def custom_init():
    with open("light_up_init.txt", encoding="utf-8") as init_file:
        board = []
        for line in init_file:
            line = line.strip()
            board.append(line.split(" "))
    # print("board: ", board)
    return board

def main():
    
    init = Light_Up(custom_init())
    board = Solution(init.board, [], init.black_cell_with_number)
    print(init)
    
    
    start_time_BFS = time.time()
    breadth_first_search(board)
    end_time_BFS = time.time()
    print("BFS: ", end_time_BFS - start_time_BFS)
    
    # start_time_heuristic = time.time()
    # best_first_search(board)
    # end_time_heuristic = time.time()
    # print("Heuristic: ", end_time_heuristic - start_time_heuristic)
    
    global result
    with open("light_up_result.txt", "w") as output_file:
        output_file.write(result)

main()