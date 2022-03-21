import string


class Light_Up:
    def __init__(self, board):
        if isinstance(board, str):
            self.board = []
            for c in board:
                if c > "1" and c < "9": # Black block with number
                    self.board.append(int(c))
                elif c == "B": # Black block without number
                    self.board.append(-1)
                else:
                    self.board += []

        else:
            pass
    


def online_init():
    level = -1
    while (level not in range(0, 12)):
        print("Input your option:")
        print("0:   7x7 Light Up Easy")
        print("1:   7x7 Light Up Normal")
        print("2:   7x7 Light Up Hard")
        print("3:   10x10 Light Up Easy")
        print("4:   10x10 Light Up Normal")
        print("5:   10x10 Light Up Hard")
        print("6:   14x14 Light Up Easy")
        print("7:   14x14 Light Up Normal")
        print("8:   14x14 Light Up Hard")
        print("9:   25x25 Light Up Easy")
        print("10:  25x25 Light Up Normal")
        print("11:  25x25 Light Up Hard")
        level = int(input())
    import requests
    # Take task string from web page
    request = requests.get(f"https://www.puzzle-light-up.com/?size={level}")
    task = request.text[request.text.find("var task ="):]
    task = task[task.find("'") + 1: task.find(";") - 1]
    return task

print(online_init())