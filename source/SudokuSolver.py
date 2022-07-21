class SudokuSolver:
    def __init__(self) -> None:
        pass

    def find_empty(self, board: list[list[int]]):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)  # row, col

        return None

    def is_valid(self, board: list[list[int]], number, position):
        # Check row
        for i in range(len(board[0])):
            if board[position[0]][i] == number and position[1] != i:
                return False

        # Check column
        for i in range(len(board)):
            if board[i][position[1]] == number and position[0] != i:
                return False

        # Check box
        box_x = position[1] // 3
        box_y = position[0] // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if board[i][j] == number and (i, j) != position:
                    return False

        return True

    def solve(self, board: list[list[int]]):
        find = self.find_empty(board)
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if self.is_valid(board, i, (row, col)):
                board[row][col] = i

                if self.solve(board):
                    return True

                board[row][col] = 0

        return False

    def print(self, board: list[list[int]]):
        print(" ┌─────────┬─────────┬────────┐")
        for i in range(len(board)):
            if i % 3 == 0 and i != 0:
                print(" ├─────────┼─────────┼────────┤ ")

            for j in range(len(board[0])):
                if j % 3 == 0:
                    print(" | ", end=" ")

                if j == 8:
                    print(str(board[i][j]) + " | ")
                else:
                    print(str(board[i][j]), end=" ")
        print(" └─────────┴─────────┴────────┘")


if __name__ == "__main__":

    grid = [
        [8, 0, 0, 0, 1, 0, 0, 0, 9],
        [0, 5, 0, 8, 0, 7, 0, 1, 0],
        [0, 0, 4, 0, 9, 0, 7, 0, 0],
        [0, 6, 0, 7, 0, 1, 0, 2, 0],
        [5, 0, 8, 0, 6, 0, 1, 0, 7],
        [0, 1, 0, 5, 0, 2, 0, 9, 0],
        [0, 0, 7, 0, 4, 0, 6, 0, 0],
        [0, 8, 0, 3, 0, 9, 0, 4, 0],
        [3, 0, 0, 0, 5, 0, 0, 0, 8],
    ]

    solver = SudokuSolver()

    # solver.print_board(grid)
    solver.solve(grid)
    # print()
    solver.print(grid)