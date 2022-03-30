def heuristic_function():
    res = 0
    for row in range(0, board.size):
        for col in range(0, board[i].size):
            if board[row][col] == 'not light up':
                res = res + 1
            elif board[row][col] in [1, 2, 3, 4] and board[row[col] != "non-number black cell":
                res = res + count_remaining_bulbs(row, col, board)
    return res