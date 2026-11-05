# Chapter 1: NumPy arrays

Python lists can hold anything, which makes them flexible but slow for numbers. NumPy gives you the **array**: a block of numbers that all share one type, stored side by side in memory. Almost every data library you will meet (pandas, Matplotlib, SciPy, scikit-learn, PyTorch) is built on top of it.

By the end of this chapter you will be able to create arrays in several ways, read their shape and type, change their shape, and explain why an array is faster than a list.

```python
import numpy as np

print("NumPy version:", np.__version__)
rng = np.random.default_rng(42)  # seeded, so every run gives the same numbers
```

## 1.1 Why not just use lists?

A list stores pointers to separate Python objects scattered around memory. To add two lists you need a loop, and every step of that loop is interpreted Python. An array stores raw numbers in one contiguous block, and the loop runs inside compiled C code. You write one expression; NumPy does the looping.

```python
prices = [120, 160, 180, 200]
qty = [2, 1, 3, 1]

# with lists you need a loop (or a comprehension)
totals_list = [p * q for p, q in zip(prices, qty)]

# with arrays the maths is element by element automatically
totals_arr = np.array(prices) * np.array(qty)

print(totals_list)
print(totals_arr)
```

Both give the same totals, but the array version reads like maths and scales to millions of values. Let's measure the difference on a million numbers.

```python
import time

n = 1_000_000
a_list = list(range(n))
a_arr = np.arange(n)

t0 = time.perf_counter(); squares_list = [x * x for x in a_list]; t_list = time.perf_counter() - t0
t0 = time.perf_counter(); squares_arr = a_arr * a_arr;            t_arr = time.perf_counter() - t0

assert squares_list[-1] == squares_arr[-1]
print(f"list: {t_list*1000:.1f} ms   array: {t_arr*1000:.1f} ms   speed-up: {t_list/t_arr:.0f}x")
```

The exact numbers depend on your machine, but the array is usually tens of times faster. This style of writing whole-array expressions instead of loops is called **vectorisation**, and Chapter 3 is all about it.

## 1.2 Creating arrays from data

`np.array()` turns a list (or a list of lists) into an array. A list of lists becomes a 2-D array: rows and columns, like a small table.

```python
a = np.array([3, 1, 4, 1, 5])
m = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a)
print(m)
```

Every array has a few attributes you will check constantly:

| Attribute | Meaning |
|---|---|
| `ndim` | number of dimensions (axes) |
| `shape` | size along each axis, as a tuple |
| `size` | total number of elements |
| `dtype` | the type shared by every element |

```python
for name, arr in [("a", a), ("m", m)]:
    print(f"{name}: ndim={arr.ndim}, shape={arr.shape}, size={arr.size}, dtype={arr.dtype}")
```

`m.shape` is `(2, 3)`: 2 rows and 3 columns. The first number is always axis 0 (rows), the second is axis 1 (columns).

## 1.3 Creating arrays from scratch

Often you need an array of a given shape before you have the data. NumPy has a constructor for each common case.

```python
print(np.zeros(4))             # four 0.0 values
print(np.ones((2, 3)))         # 2 rows x 3 columns of 1.0
print(np.full((2, 2), 7))      # every value is 7
print(np.eye(3))               # identity matrix
```

For sequences, `np.arange` works like `range` but returns an array, and `np.linspace` gives a fixed **number** of evenly spaced points, including both ends.

```python
print(np.arange(0, 10, 2))      # start, stop (excluded), step
print(np.linspace(0, 1, 5))     # start, stop (included), how many points
```

Use `arange` when you know the step and `linspace` when you know how many points you want. With fractional steps prefer `linspace`: floating-point rounding can make `arange` return one value more or less than you expect.

## 1.4 Random arrays

Random numbers are everywhere in data work: simulations, sampling, splitting data. Modern NumPy uses a **Generator** object, which we created in the setup cell as `rng`. Seeding it means your results are reproducible.

```python
print(rng.integers(1, 7, size=10))          # ten dice rolls (7 is excluded)
print(rng.random(3))                         # uniform in [0, 1)
print(rng.normal(loc=170, scale=8, size=4))  # heights in cm, mean 170, sd 8
print(rng.choice(["UPI", "Card", "Cash"], size=5, p=[0.6, 0.25, 0.15]))
```

You may see `np.random.seed(0)` and `np.random.rand()` in older tutorials. That legacy interface still works, but `default_rng` is the recommended way: it is faster, has better statistical properties and keeps its state in an object you can pass around.

## 1.5 Data types (dtype)

All elements of an array share one dtype. NumPy picks one from your data, and you can choose it with `dtype=`.

```python
print(np.array([1, 2, 3]).dtype)          # whole numbers -> int64
print(np.array([1, 2, 3.5]).dtype)        # one float makes everything float64
print(np.array([True, False]).dtype)      # bool
print(np.array([1, 2, 3], dtype=np.float32).dtype)
```

Because elements share a type, mixing types **upcasts**: one float turns the whole array into floats. Use `astype` to convert, and remember that converting floats to integers **truncates** (it chops off the decimals rather than rounding).

```python
x = np.array([1.9, -1.9, 2.5])
print(x.astype(int))            # truncation toward zero
print(np.round(x).astype(int))  # round first if that is what you mean
```

Smaller dtypes save memory, which matters for big arrays:

```python
big64 = np.ones(1_000_000, dtype=np.float64)
big32 = np.ones(1_000_000, dtype=np.float32)
print(f"float64: {big64.nbytes/1e6:.0f} MB, float32: {big32.nbytes/1e6:.0f} MB")
```

Integers can also overflow silently when the type is too small. An `int8` holds -128 to 127:

```python
small = np.array([120, 125], dtype=np.int8)
print(small + 10)   # wraps around instead of reaching 130
```

## 1.6 Reshaping

`reshape` gives the same data a new shape. The total number of elements must stay the same. Pass `-1` for one dimension and NumPy will work it out.

```python
x = np.arange(12)
print(x.reshape(3, 4))
print(x.reshape(2, -1).shape)   # -1 becomes 6
```

`ravel()` flattens back to 1-D, and `.T` transposes (rows become columns).

```python
grid = np.arange(6).reshape(2, 3)
print(grid.T)
print(grid.ravel())
```

You will often need to turn a 1-D array into a single column or a single row, for example before feeding data to scikit-learn in the machine learning stage. `np.newaxis` (or `reshape`) adds an axis of length 1:

```python
v = np.array([10, 20, 30])
print(v[:, np.newaxis].shape)   # column: (3, 1)
print(v[np.newaxis, :].shape)   # row:    (1, 3)
```

## 1.7 Combining and splitting arrays

`np.concatenate` joins arrays along an existing axis. `np.stack` creates a **new** axis, which is handy for turning several 1-D arrays into the columns of a table.

```python
mon = np.array([12, 15, 11])
tue = np.array([14, 13, 16])

print(np.concatenate([mon, tue]))         # one long 1-D array
print(np.stack([mon, tue]))               # 2 rows (one per day)
print(np.column_stack([mon, tue]))        # 2 columns (one per day)
```

To go the other way, `np.split` cuts an array into equal pieces, or at the positions you give it.

```python
parts = np.split(np.arange(9), 3)
print(parts)
print(np.split(np.arange(9), [2, 5]))     # cut before index 2 and before index 5
```

## 1.8 Saving and loading arrays

NumPy can write arrays to disk in its own binary format (`.npy`, or `.npz` for several arrays at once). It keeps the dtype and shape exactly and loads very fast. For data to share with other tools, use CSV with `np.savetxt`, or better, pandas (Chapter 6).

```python
import tempfile, os

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "grid.npy")
    np.save(path, grid)
    back = np.load(path)
    assert np.array_equal(back, grid)
    print(back, back.dtype)
```

## 1.9 Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Expecting `list * 2` to double numbers | A list repeats itself: `[1, 2, 1, 2]` | Convert first: `np.array(lst) * 2` |
| Mixing strings and numbers in one array | Everything becomes text (dtype `<U...`) and maths fails | Keep one type per array; use pandas for mixed tables |
| `astype(int)` to round | Decimals are truncated, so 1.9 becomes 1 | `np.round(x).astype(int)` |
| Small integer dtype for large values | Silent overflow wraps around | Use `int64` or `float64` unless memory is tight |
| `np.arange(0, 1, 0.1)` for exact points | Rounding can add or drop the last point | `np.linspace(0, 1, 11)` |
| Reshape with the wrong total size | `ValueError: cannot reshape` | Check `arr.size`; use `-1` for one axis |

## 1.10 Cheat sheet

| Task | Code |
|---|---|
| Array from a list | `np.array([1, 2, 3])` |
| Zeros, ones, constant | `np.zeros((2, 3))`, `np.ones(5)`, `np.full((2, 2), 7)` |
| Range with a step | `np.arange(0, 10, 2)` |
| N evenly spaced points | `np.linspace(0, 1, 50)` |
| Seeded random generator | `rng = np.random.default_rng(42)` |
| Random integers / normal | `rng.integers(0, 10, 5)`, `rng.normal(0, 1, 5)` |
| Shape, dims, size, type | `a.shape`, `a.ndim`, `a.size`, `a.dtype` |
| Convert type | `a.astype(np.float32)` |
| Reshape (infer one axis) | `a.reshape(3, -1)` |
| Flatten / transpose | `a.ravel()`, `a.T` |
| 1-D to column | `a[:, np.newaxis]` or `a.reshape(-1, 1)` |
| Join / stack | `np.concatenate([a, b])`, `np.column_stack([a, b])` |
| Save / load | `np.save("a.npy", a)`, `np.load("a.npy")` |

## 1.11 Exercises

**1.** Create an array of the 12 monthly sales targets `[50, 55, ..., 105]` (start at 50, step 5) and reshape it into 4 quarters x 3 months.

```python
# Solution
targets = np.arange(50, 110, 5)
quarters = targets.reshape(4, 3)
print(quarters)
```

**2.** Simulate 1,000 rolls of two dice with `rng` and store them as an array of shape `(1000, 2)`. Then print the first five rows and the dtype.

```python
# Solution
rolls = rng.integers(1, 7, size=(1000, 2))
print(rolls[:5])
print(rolls.shape, rolls.dtype)
```

**3.** You have temperatures `[21.7, 22.4, 19.5, 23.5]`. Make an integer array that **rounds** them correctly, and check the result with `assert`.

```python
# Solution
temps = np.array([21.7, 22.4, 19.5, 23.5])
rounded = np.round(temps).astype(int)
print(rounded)
assert rounded.tolist() == [22, 22, 20, 24]  # NumPy rounds .5 to the nearest even number
```

Note the last line: `np.round` uses "round half to even" (19.5 becomes 20, but 22.5 would become 22). This avoids a bias when you round many values.

**4.** Stack the three arrays `height = [160, 172, 181]`, `weight = [55, 70, 82]` and `age = [24, 31, 45]` into a table with one row per person and one column per measurement. What is its shape?

```python
# Solution
height = np.array([160, 172, 181])
weight = np.array([55, 70, 82])
age = np.array([24, 31, 45])
people = np.column_stack([height, weight, age])
print(people)
print(people.shape)  # (3 people, 3 measurements)
```
