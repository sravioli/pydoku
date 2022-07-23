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
 │  2 0 5  │  0 0 7  │  0 0 6 |
 │  4 0 0  │  9 6 0  │  0 2 0 |
 │  0 0 0  │  0 8 0  │  0 4 5 |
 ├─────────┼─────────┼────────┤
 │  9 8 0  │  0 7 4  │  0 0 0 |
 │  5 7 0  │  8 0 2  │  0 6 9 |
 │  0 0 0  │  6 3 0  │  0 5 7 |
 ├─────────┼─────────┼────────┤
 │  7 5 0  │  0 2 0  │  0 0 0 |
 │  0 6 0  │  0 5 0  │  0 0 2 |
 │  3 0 0  │  4 0 0  │  5 0 8 |
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
