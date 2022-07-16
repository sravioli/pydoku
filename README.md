# Pydoku

Creating an automatic Sudoku puzzle solver with OpenCV is a 6-step process:

- Step #1: Provide input image containing Sudoku puzzle to our system.
- Step #2: Locate where in the input image the puzzle is and extract the board.
- Step #3: Given the board, locate each of the individual cells of the Sudoku board (most standard Sudoku puzzles are a 9×9 grid, so we’ll need to localize each of these cells).
- Step #4: Determine if a digit exists in the cell, and if so, OCR it.
- Step #5: Apply a Sudoku puzzle solver/checker algorithm to validate the puzzle.
- Step #6: Display the output result to the user.

The biggest exception is Step #4, where we need to apply OCR.
OCR can be a bit tricky to apply, but we have a number of options:

- Use the Tesseract OCR engine, the de facto standard for open source OCR
- Utilize cloud-based OCR APIs, such as Microsoft Cognitive Services, Amazon Rekognition, or the Google Vision API
- Train our own custom OCR model
