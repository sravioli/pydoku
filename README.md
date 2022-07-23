# Pydoku

Pydoku is a sudoku solver in OpenCV, TensorFlow and Keras.

## Get started

First clone the repository:

- With GitHub CLI:

  ```shell
  $ gh repo clone sRavioli/pydoku
  Cloning into 'pydoku'...
  ```

- With Git:

  ```shell
  $ git clone https://github.com/sRavioli/pydoku
  Cloning into 'pydoku'...
  ```

Then `cd` to the directory. Install the required dependencies:

```shell
pip install -r requirements.txt
```

or clone the conda environment:

```shell
conda env create -f pydoku.yaml
```

Activate the conda environment, then run the app

```shell
$ ~/.conda/envs/pydoku/python.exe ./app.py

 ┌─────────┬─────────┬────────┐
 │  2 1 5  │  3 4 7  │  9 8 6 |
 │  4 3 8  │  9 6 5  │  7 2 1 |
 │  6 9 7  │  2 8 1  │  3 4 5 |
 ├─────────┼─────────┼────────┤
 │  9 8 6  │  5 7 4  │  2 1 3 |
 │  5 7 3  │  8 1 2  │  4 6 9 |
 │  1 4 2  │  6 3 9  │  8 5 7 |
 ├─────────┼─────────┼────────┤
 │  7 5 9  │  1 2 8  │  6 3 4 |
 │  8 6 4  │  7 5 3  │  1 9 2 |
 │  3 2 1  │  4 9 6  │  5 7 8 |
 └─────────┴─────────┴────────┘
```

To change the sudoku image, open `app.py` and edit, on line 23:

```python3
# import image
original_image = ipr.read("./source/test_imgs/1.jpg")
```

and insert the relative path of the image.

### Known limitations

- Upon warping and cropping an already flat image, the latter gets rotated
  clockwise by $90^\circ$;
- Upon determining whether a grid cell contains a number, the algorithm will
  sometimes consider cell containing a number empty. This is why the function
  `sxt.construct_board()` accepts an optional argument: the minimum number of
  white pixels;
- Upon predicting the digits, the CNN will sometimes confuse the number $1$ with
  the number $7$, rarely the number $6$ with number $5$ or $8$. This is probably
  due to the extraction process of the digit.

## License

The content of the project itself is licensed under the [GNU General Public License v3.0](https://github.com/sRavioli/pydoku/blob/main/LICENCE.txt).
