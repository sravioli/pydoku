class SudokuSolver:
    def __init__(self) -> None:
        pass

    def find_empty(self, board: list[list[int]]) -> tuple[int, int]:
        """Finds an empty board cell.

        Args:
            board (list[list[int]]): The sudoku board from which to find an empty cell.

        Returns:
            tuple[int, int]: The coordinates of the empty cell.
        """
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)  # row, col

        return None

    def is_valid(
        self,
        board: list[list[int]],
        number: int,
        coords: tuple[int, int],
    ) -> bool:
        """Checks if the given number can be inserted in the given coordinates.

        Args:
            board (list[list[int]]): The sudoku board.
            number (int): A number ranging from 0 to 8.
            coords (tuple[int, int]): The coordinates of the cell.

        Returns:
            bool: Whether or not the given number can be inserted in the given
                coordinates.
        """
        # Check row
        for i in range(len(board[0])):
            if board[coords[0]][i] == number and coords[1] != i:
                return False

        # Check column
        for i in range(len(board)):
            if board[i][coords[1]] == number and coords[0] != i:
                return False

        # Check box
        box_x = coords[1] // 3
        box_y = coords[0] // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if board[i][j] == number and (i, j) != coords:
                    return False

        return True

    def solve(self, board: list[list[int]]) -> None:
        """Solves the given sudoku.

        Args:
            board (list[list[int]]): The sudoku board.

        Returns:
            None: The function modifies the original board.
        """
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

    def print(self, board: list[list[int]]) -> str:
        """Prints the sudoku board with grid lines.

        Args:
            board (list[list[int]]): The sudoku board.

        Returns:
            str: Prints the sudoku board with grid lines.
        """
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


# for debugging purposes
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
