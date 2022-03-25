from enum import Enum
from copy import deepcopy
from itertools import count
from re import X
import time


count1 = 0
count2 = 0
count3 = 0
count4 = 0


block=["0","1","2","3","4","B"]
class Node:
    board=[]
    child=[]
    numberedcell=[]
    def __init__(self,board,child,numberedcell):
        self.board=board
        self.child=child
        self.numberedcell=numberedcell
    
    def isgoal(self): 
        for i in self.board:
            for j in i:
                if j == "-":
                    return False
        if self.checkallblackcell()==False:
            return False
        return True

    def checkallblackcell(self):
        for i in range (0,7):
            for j in range (0,7):
                if (self.board[i][j] in block) and (self.board[i][j] !="B"):
                    if self.checkoneblackcell(i,j,int(self.board[i][j]))!=0:
                        return False    # the number of bulbs != the number in the black cell
        return True

    def checkoneblackcell(self,i,j,t): # check 4 corners of the cell to find bulbs
        list=[]
        if i>0:
            list.append(self.board[i-1][j])
        if i<6:
            list.append(self.board[i+1][j])
        if j>0:
            list.append(self.board[i][j-1])
        if j<6:
            list.append(self.board[i][j+1])
        return abs(t-list.count("x"))   # cuz bulbs can be more or less than the number in black cell ---> abs

    #kiểm tra 1 ô đen đc đánh 0,1,2,3 có nhiêu bóng đèn xung quanh nó
    def blackcellnotexceed(self,i,j,t): # check the black cell is having how many bulbs around 4 corners
        list=[]
        if i>0:
            list.append(self.board[i-1][j])
        if i<6:
            list.append(self.board[i+1][j])
        if j>0:
            list.append(self.board[i][j-1])
        if j<6:
            list.append(self.board[i][j+1])
        return t-list.count("x")    # check the number of bulbs that can be filled more, if < 0 --> false
    
    # khi đặt 1 bóng đèn vào ô (i,j) số bóng đèn xung quanh ở các ô đen chung cạnh ô (i,j) sẽ thay đổi
    # ta cần phải kiểm tra số bóng đèn xung quanh của 4 ô đen chung cạnh ô (i,j)
    def blackcellnotviolated(self,i,j):
        if i>0 and (self.board[i-1][j] in block) and (self.board[i-1][j] !="B"):
            if self.blackcellnotexceed(i-1,j,int(self.board[i-1][j]))<0:
                return False
        if i<6 and (self.board[i+1][j] in block) and (self.board[i+1][j] !="B"):
            if self.blackcellnotexceed(i+1,j,int(self.board[i+1][j]))<0:
                return False
        if j>0 and (self.board[i][j-1] in block) and (self.board[i][j-1] !="B"):
            if self.blackcellnotexceed(i,j-1,int(self.board[i][j-1]))<0:
                return False
        if j<6 and (self.board[i][j+1] in block) and (self.board[i][j+1] !="B"):
            if self.blackcellnotexceed(i,j+1,int(self.board[i][j+1]))<0:
                return False
        return True

    def placelamp(self,i,j): 
        m=deepcopy(self.board)
        m[i][j]="x"
        t=i-1
        # light up the column from the cell 'x' (placed bulb) to above
        while t >=0 and self.board[t][j] not in block:
            m[t][j]="l"
            t=t-1
        
        # light up the column from the cell 'x' (placed bulb) to bottom
        t=i+1
        while t <=6 and self.board[t][j] not in block:
            m[t][j]="l"
            t=t+1
            
        # light up the row from the cell 'x' (placed bulb) to left
        t=j-1
        while t >=0 and self.board[i][t] not in block:
            m[i][t]="l"
            t=t-1
        
        # light up the row from the cell 'x' (placed bulb) to right
        t=j+1
        while t <=6 and self.board[i][t] not in block:
            m[i][t]="l"
            t=t+1
            
        return m    # return new board after placed bulb and light up
    
    # xóa từ danh sách numberedcell ô đc đánh 1,2,3 mà đã đủ bóng đèn xung quanh nó
    def removesastisfiedblackcell(self):
        i=0
        while i<len(self.numberedcell):     # check all black cell with number 1,2,3
            m=self.numberedcell[i][0]   # row
            n=self.numberedcell[i][1]   # col
            p=self.blackcellnotexceed(m,n,int(self.board[m][n]))
            if p==0:    # completely filled all bulbs that satisfy condition number of the black cell
                self.numberedcell.pop(i)
            i=i+1


    

    def generatechild(self):
        childlist=[]
        a=deepcopy(self.numberedcell)

        global count1
        
        
        m=deepcopy(self)
        
        
        # if count1 != 1:
        #     count1 +=1
        #     print("a: ", a)
        #     print("m: ", m.numberedcell)
        
        
        # ta sẽ ưu tiên thăm các ô đen đc đánh số 1,2,3 mà ô đc đánh số i có ÍT HƠN i bóng đèn xung quanh
        # a chính là list những ô như vậy
        # Nếu a rỗng, khi này mọi ô đen đc đánh 1,2,3 đều đã đủ các bóng đèn xung quanh nó
        # Khi này chọn ô trắng đầu tiên chưa đc chiếu sáng, ta visit tất cả các ô mà chiếu sáng ô trắng này
        # như vậy trường hợp xấu nhất ta tạo 13 child (là những ô chung hàng và cột) 
        if a==[]:
            found=False
            for i in range (0,7):
                for j in range (0,7):
                    if self.board[i][j] == "-":
                        b=m.placelamp(i,j)
                        c=Node(b,self.child,a)
                        z=c.blackcellnotviolated(i,j)
                        if z==True:
                                childlist.append(c)
                        t=i-1
                        while t >=0 and self.board[t][j] not in block:
                            if self.board[t][j]=="-":
                                b=m.placelamp(t,j)
                                c=Node(b,self.child,a)
                                z=c.blackcellnotviolated(t,j)
                                if z==True:
                                    childlist.append(c)
                            t=t-1
                        t=i+1
                        while t <=6 and self.board[t][j] not in block:
                            if self.board[t][j]=="-":
                                b=m.placelamp(t,j)
                                c=Node(b,self.child,a)
                                z=c.blackcellnotviolated(t,j)
                                if z==True:
                                    childlist.append(c)
                            t=t+1
                        t=j-1
                        while t >=0 and self.board[i][t] not in block:
                            if self.board[i][t]=="-":
                                b=m.placelamp(i,t)
                                c=Node(b,self.child,a)
                                z=c.blackcellnotviolated(i,t)
                                if z==True:
                                    childlist.append(c)
                            t=t-1
                        t=j+1
                        while t <=6 and self.board[i][t] not in block:
                            if self.board[i][t]=="-":
                                b=m.placelamp(i,t)
                                c=Node(b,self.child,a)
                                z=c.blackcellnotviolated(i,t)
                                if z==True:
                                    childlist.append(c)  
                            t=t+1
                        break
                if found==True:
                    break
        # nếu a khác rỗng, khi này sẽ có 1 ô đen chưa đủ bóng đèn xung quanh nó, khi này ta
        # sẽ visit 4 ô xung quanh ô đen, vậy trường hợp xấu nhất ta tạo ra 4 child
        else:
            # example: a = [ [1, 4], [2, 1] ...]; 1 and 4 is index of row and column of black cell
            n=a[0]  # get the first pair [row, col]
            n1=n[0] # row
            n2=n[1] # col
            
            global count3
            if count3 != 3:
                count3 +=1
                print("row, col: ", n1, n2)
            
            if n1-1>=0 and self.board[n1-1][n2]=="-":
                b=m.placelamp(n1-1,n2)  # m is root (board, child, numberedcell) ---> return a board (b)
                c=Node(b,self.child,a)
                z=c.blackcellnotviolated(n1-1,n2)            
                if z==True:
                    c.removesastisfiedblackcell()
                    childlist.append(c)
            if n1+1<=6 and self.board[n1+1][n2]=="-":
                b=m.placelamp(n1+1,n2)
                c=Node(b,self.child,a)
                z=c.blackcellnotviolated(n1+1,n2)            
                if z==True:
                    c.removesastisfiedblackcell()
                    childlist.append(c)
            if n2-1>=0 and self.board[n1][n2-1]=="-":
                b=m.placelamp(n1,n2-1)
                c=Node(b,self.child,a)
                z=c.blackcellnotviolated(n1,n2-1)            
                if z==True:
                    c.removesastisfiedblackcell()
                    childlist.append(c)  
            if n2+1<=6 and self.board[n1][n2+1]=="-":
                b=m.placelamp(n1,n2+1)
                c=Node(b,self.child,a)
                z=c.blackcellnotviolated(n1,n2+1)            
                if z==True:
                    c.removesastisfiedblackcell()
                    childlist.append(c)  
        return childlist

    def printboard(self):
        for i in range(0,7):
            for j in range(0,7):
                print(self.board[i][j],end="")
                print(" ",end="")
            print()

    
        


def heuristic(node):
        h=0
        for i in range(0,7):
            for j in range(0,7):
                if node.board[i][j]=='-':
                    h=h+1
                elif node.board[i][j] in block and node.board[i][j] !="B":
                    h=h+node.checkoneblackcell(i,j,int(node.board[i][j]))
                    
        global count4
        if count4 != 50:
            count4 +=1
            print("h: ", h)
        
        return h


def breadthfirstsearch(root):
    visitlist=[]
    queue=[]
    queue.append(root)
    visitlist.append(root.board)        # board: [ [], [] ...] ---> append: [ [ [], [] ...] ]
    
    # print("root: ", root)   
    # print("queue: ", queue)
    # print("visit list: ", visitlist)
    
    
    while queue and (queue[0].isgoal()==False):
        a=queue.pop(0)
        a.child=a.generatechild()
        a.printboard()
        print()
        print("---------------------------------")
        print()

        for i in a.child:
            if i.board not in visitlist:
                queue.append(i)
                visitlist.append(i.board)
        
        # global count2
        # if count2 != 3:
        #     count2 +=1
        #     # for i in a.child:
        #     #     print("child board: ", i.board)
        #     #     print("child numberedcell: ", i.numberedcell)
        #     #     print()
        #     print("queue: ")
        #     for i in range(len(queue)):
        #         print("queue[", i, "]: ")
        #         # for j in range(len(queue[i].board)):
        #         #     print(queue[i].board[j])
        #         queue[i].printboard()
        #         print()
        #     # print("a child board: ", a.child.numberedcell)
        
        
    queue[0].printboard()

#cũng giống breadth first nhưng sort queue sau mỗi lần thêm
def bestfirstsearch(root):
    visitlist=[]
    queue=[]
    queue.append(root)
    
    
    
    visitlist.append(root.board)
    while queue and (queue[0].isgoal()==False):
        # a is root (board, child, numberedcell)
        a=queue.pop(0)
        a.child=a.generatechild()
        a.child.sort(key=heuristic)
        print()
        a.child.sort(key=heuristic)
        
        
        
        print()
        
        
        
        a.printboard()
        print()
        print("---------------------------------")
        print()
        for i in a.child:
            if i.board not in visitlist:
                queue.append(i)
                visitlist.append(i.board)
        queue.sort(key=heuristic)
        print()
        queue.sort(key=heuristic)
        
        
        
        print()
        
        
        global count2
        if count2 != 3:
            count2 +=1
            # for i in a.child:
            #     print("child board: ", i.board)
            #     print("child numberedcell: ", i.numberedcell)
            #     print()
            print("queue: ")
            for i in range(len(queue)):
                print("queue[", i, "]: ")
                # for j in range(len(queue[i].board)):
                #     print(queue[i].board[j])
                queue[i].printboard()
                print()
            # print("a child board: ", a.child.numberedcell)
            
            
    queue[0].printboard()
        



def main():
    start_time = time.time()
    #đọc input
    d=open("Input.txt","r")
    arr=[]
    numberedcell=[]
    
    # d: 2d array
    # arr: 2d array, format: [ [], [] ...]
    for x in d:
        arr2=[]
        for i in x:
            arr2.append(i)
        arr.append(arr2)
        
    for i in range(0,len(arr)-1):
        arr[i]=arr[i][:-1]  # arr[i] contains'\n' as the last element --> remove it
        
    # tạo list gồm danh sách những ô đc đánh số 1,2,3 mà ô đc đánh số i có ÍT HƠN i bóng đèn xung quanh
    for i in range(0,7):
        for j in range(0,7):
            # only choose black cell with 1,2,3,4
            if arr[i][j] in block and arr[i][j] !="B" and arr[i][j] !="0":
                numberedcell.append([i,j])
                
    print("number cell: ", numberedcell)
    
    # for i in range(len(arr)):
    #     for j in range(len(arr[i])):
    #         print(arr[i][j] + " ")
    #     print()
    
    # print("arr: ", arr)

    board=Node(arr,[],numberedcell)
    # gọi hàm search
    bestfirstsearch(board)   
    end_time = time.time()
    elapsed_time = end_time - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

main()


