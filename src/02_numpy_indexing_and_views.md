# Chapter 2: NumPy indexing and views

Creating an array is only half the job. Most real work is about **picking out** the values you care about: the last week of sales, the rainy days, the three busiest stores. NumPy gives you a small set of tools for this, and once you know them you will use them in every chapter that follows, including all of pandas.

By the end of this chapter you will be able to slice 1-D and 2-D arrays, filter them with conditions, pick values by position lists, sort them, and explain when a result shares memory with the original. It builds on Chapter 1, especially shapes (see 1.2) and reshaping (see 1.6).

```python
import numpy as np

rng = np.random.default_rng(42)
```

## 2.1 Picking single elements

Indexing an array works like indexing a list: positions start at 0, and negative positions count from the end. `-1` is the last element, `-2` the one before it.

```python
cups = np.array([42, 38, 51, 47, 60, 72, 66])   # cups sold, Monday to Sunday

print(cups[0])     # Monday
print(cups[3])     # Thursday
print(cups[-1])    # Sunday, counted from the end
print(cups[-2])    # Saturday
```

The first and last values print as plain numbers. Asking for a position that does not exist raises an `IndexError`, exactly as with a list:

```python
try:
    cups[7]
except IndexError as e:
    print("IndexError:", e)
```

You can also change one value in place by assigning to its position:

```python
cups[2] = 50       # correct a typo in Wednesday's count
print(cups)
```

## 2.2 Slicing with start, stop and step

A **slice** `a[start:stop:step]` takes a run of elements. As with `range`, `start` is included and `stop` is excluded. Any part can be left out: a missing start means "from the beginning", a missing stop means "to the end", and a missing step means 1.

```python
week = np.arange(1, 8)          # day numbers 1..7

print(week[1:4])    # positions 1, 2, 3
print(week[:3])     # first three
print(week[4:])     # from position 4 to the end
print(week[::2])    # every second day
```

A negative step walks backwards. `a[::-1]` is the quickest way to reverse an array, and negative positions work inside slices too.

```python
print(week[::-1])     # reversed
print(week[-3:])      # last three days
print(week[5:1:-1])   # from position 5 down to (not including) position 1
```

Unlike single indices, slices never raise an error when they go past the end. They just stop early, so `week[5:100]` gives the last two days.

```python
print(week[5:100])
```

## 2.3 Rows, columns and blocks in 2-D

A 2-D array takes one index per axis, separated by a comma: `m[row, col]`. Each part can be a single number or a slice, so you can take one cell, a full row, a full column or a rectangular block.

Let's load a real table. `data/weather.csv` holds one temperature per day for three cities, listed city by city (all of Pune's 366 days, then Mumbai's, then Bengaluru's). `np.loadtxt` reads numeric columns from a text file.

```python
temp = np.loadtxt("data/weather.csv", delimiter=",", skiprows=1, usecols=2)
print(temp.shape)

cities = np.array(["Pune", "Mumbai", "Bengaluru"])
temp = temp.reshape(3, 366)      # one row per city, one column per day of 2024
print(temp.shape)
```

Now axis 0 is the city and axis 1 is the day of the year.

```python
print(temp[1, 0])          # Mumbai on 1 January (one cell)
print(temp[2, :5])         # Bengaluru, first five days (part of a row)
print(temp[:, 0])          # all three cities on 1 January (a column)
print(temp[0].size)        # leaving out the column index means "all columns"
```

`temp[0]` is the whole Pune row, all 366 values (we print only its size to keep the output short). A block is just a slice on both axes:

```python
first_week = temp[:, :7]       # every city, days 0-6
print(first_week.shape)
print(first_week)
```

The result is a smaller 2-D array of shape `(3, 7)`. Notice the difference between `temp[0]` (shape `(366,)`, a 1-D row) and `temp[0:1]` (shape `(1, 366)`, still 2-D). A single number removes an axis; a slice keeps it.

```python
print(temp[0].shape, temp[0:1].shape)
```

## 2.4 Views and copies

Here is the most surprising thing about NumPy slicing: **a slice does not copy the data**. It returns a **view**, a new array object that looks at the same block of memory. This makes slicing instant even on huge arrays, but it also means that changing the view changes the original.

```python
prices = np.array([120, 160, 180, 200, 90])
coffee = prices[:4]          # a view of the first four prices

coffee[0] = 999              # change the view...
print(prices)                # ...and the original changed too
```

`np.shares_memory` tells you whether two arrays use the same memory. When you want an independent array, call `.copy()`.

```python
prices = np.array([120, 160, 180, 200, 90])
view = prices[:4]
safe = prices[:4].copy()

print(np.shares_memory(prices, view))   # True
print(np.shares_memory(prices, safe))   # False

safe[0] = 999
print(prices)                           # unchanged
```

Reshaping and transposing also usually return views (see 1.6). The indexing tools in the rest of this chapter (boolean masks and position lists) are different: they always build a **new** array, i.e. a copy.

| Operation | Result |
|---|---|
| Basic slice `a[2:5]`, `m[:, 0]` | view |
| `reshape`, `.T`, `ravel` (usually) | view |
| Boolean mask `a[a > 0]` | copy |
| Position list `a[[0, 2]]` | copy |
| `a.copy()` | copy |

Why care? Two reasons. First, a function that modifies a slice it was given may silently change the caller's data. Second, a small slice of a huge array keeps the whole huge array alive in memory. If in doubt, `.copy()`.

```python
big = np.zeros(10_000_000)
small = big[:10].copy()        # without .copy(), 'small' would keep all 80 MB alive
print(small.nbytes, "bytes")
```

## 2.5 Boolean masks and combining conditions

Comparing an array with a value gives an array of `True`/`False`, one per element. This is called a **mask**. Putting a mask inside square brackets keeps only the positions where the mask is `True`. This is how you filter data in NumPy, and it is exactly how pandas filtering works too (Chapter 6).

```python
cups = np.array([42, 38, 50, 47, 60, 72, 66])
busy = cups > 55

print(busy)
print(cups[busy])          # only the busy days
print(busy.sum())          # True counts as 1, so this counts busy days
```

Summing a mask is a handy way to count matches, because `True` counts as 1 and `False` as 0. Let's ask a real question: how many days in 2024 was Pune warmer than 30 °C?

```python
pune = temp[0]
print((pune > 30).sum(), "days above 30 °C in Pune")
print(pune[pune > 30][:5])     # the first five of those temperatures
```

To combine conditions, use `&` (and), `|` (or) and `~` (not). Python's own `and`, `or` and `not` do **not** work on arrays. Each condition must be wrapped in parentheses, because `&` and `|` bind more tightly than `>` and `<`.

```python
rain = np.loadtxt("data/weather.csv", delimiter=",", skiprows=1, usecols=3).reshape(3, 366)
pune_rain = rain[0]

warm_and_dry = (pune > 25) & (pune_rain == 0)
cold_or_wet = (pune < 18) | (pune_rain > 20)

print("warm and dry days:", warm_and_dry.sum())
print("cold or very wet days:", cold_or_wet.sum())
print("not warm-and-dry:", (~warm_and_dry).sum())
```

The first and last counts are opposites: "warm and dry" plus "not warm and dry" always adds up to 366. Forgetting the parentheses is the most common error here; see 2.9.

## 2.6 Assigning through masks, and np.where

A mask can also appear on the **left** of `=`. Then it changes only the matching positions, in the original array. This is the NumPy way to say "for every value that ..., set it to ...".

```python
readings = np.array([21.5, -999.0, 23.1, 22.8, -999.0, 24.0])   # -999 means "sensor failed"
readings[readings == -999] = np.nan                              # mark as missing
print(readings)
```

Note that `np.nan` only fits in a float array. In an integer array you would need a different placeholder, or convert with `astype(float)` first (see 1.5).

Capping values works the same way. Here no order may contain more than 10 units:

```python
qty = np.array([1, 2, 40, 3, 1, 12])
qty[qty > 10] = 10
print(qty)
```

When you want a **new** array that picks between two options, use `np.where(condition, value_if_true, value_if_false)`. It works element by element, like an `if`/`else` applied to every position at once.

```python
cups = np.array([42, 38, 50, 47, 60, 72, 66])
label = np.where(cups > 55, "busy", "normal")
print(label)

bonus = np.where(cups > 55, cups * 0.1, 0)    # 10% bonus points on busy days
print(bonus)
```

With only a condition, `np.where(mask)` returns the **positions** where the mask is `True`. For a 1-D array you take element `[0]` of the result to get them as an array.

```python
hot_days = np.where(pune > 31.5)[0]
print(hot_days)          # day-of-year positions (0 = 1 January)
```

## 2.7 Fancy indexing with position lists

Sometimes you know exactly which positions you want, and they are not a neat slice. Pass a list (or array) of integers and NumPy returns those elements in that order. This is called **fancy indexing** or integer-array indexing.

```python
menu_prices = np.array([120, 160, 180, 200, 90, 110, 140, 110, 220, 70])
order = [2, 2, 6, 9]          # positions of the items one customer bought

print(menu_prices[order])          # repeats are allowed
print(menu_prices[order].sum())    # bill total
```

Order matters and repeats are allowed, which makes fancy indexing a lookup tool: turn codes into values. On a 2-D array, a list in one position picks whole rows or columns:

```python
print(temp[[2, 0], :3])      # Bengaluru then Pune, first three days
print(temp[:, [0, 182, 365]].round(1))  # all cities on 1 Jan, 1 Jul and 31 Dec
```

Remember that fancy indexing always returns a copy (see 2.4). Assigning **through** it, as in `a[[0, 3]] = 0`, does change the original, just like with masks.

```python
a = np.arange(6)
picked = a[[0, 3]]
picked[:] = -1               # changes only the copy
a[[1, 4]] = -1               # changes a itself
print(a, picked)
```

## 2.8 Sorting, argsort and argmax

`np.sort(a)` returns a sorted copy. Often, though, you need to know **where** values came from, not just the values. `np.argsort(a)` returns the positions that would sort the array; you can then use those positions (fancy indexing again) to reorder other arrays the same way.

```python
items = np.array(["Espresso", "Latte", "Masala Chai", "Croissant", "Cookie"])
sold = np.array([310, 540, 620, 280, 150])

print(np.sort(sold))                 # just the values, smallest first
order = np.argsort(sold)[::-1]       # positions, largest first
print(order)
print(items[order])                  # names, best seller first
```

Reversing the argsort result with `[::-1]` (see 2.2) gives descending order. `argmax` and `argmin` return the position of the single largest or smallest value, which answers questions like "which day was hottest?".

```python
hottest = pune.argmax()
print("hottest day of year:", hottest, "->", pune[hottest], "°C")

city_means = temp.mean(axis=1)            # one mean per city (axis is explained in 3.3)
print(cities[city_means.argmax()], "is the warmest city on average")
```

To find the top few values without sorting everything, combine `argsort` with a slice:

```python
top3 = np.argsort(pune)[-3:][::-1]
print(top3, pune[top3])
```

## 2.9 Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Changing a slice and expecting the original to stay the same | The original changes too, because a slice is a view | Use `a[2:5].copy()` when you need independent data |
| `a > 5 & a < 10` without parentheses | `&` runs first, giving a confusing error or wrong result | `(a > 5) & (a < 10)` |
| Using `and` / `or` with arrays | `ValueError: The truth value of an array ... is ambiguous` | Use `&`, <code>&#124;</code>, `~` |
| `m[0][1]` in a loop over 2-D data | Works but builds a temporary row each time | Use one index tuple: `m[0, 1]` |
| Assigning `np.nan` into an integer array | `ValueError: cannot convert float NaN to integer` | Convert first: `a = a.astype(float)` |
| Expecting `np.sort` or `argsort` to give descending order | Results are always ascending | Reverse with `[::-1]` |

## 2.10 Cheat sheet

| Task | Code |
|---|---|
| First / last element | `a[0]`, `a[-1]` |
| Slice (stop excluded) | `a[2:5]` |
| Every second element / reverse | `a[::2]`, `a[::-1]` |
| One cell of a 2-D array | `m[row, col]` |
| A row / a column | `m[0]`, `m[:, 0]` |
| A block | `m[:2, 1:4]` |
| Independent copy | `b = a[2:5].copy()` |
| Do two arrays share memory? | `np.shares_memory(a, b)` |
| Filter by a condition | `a[a > 10]` |
| Combine conditions | `a[(a > 10) & (a < 20)]`, <code>&#124;</code> for or, `~` for not |
| Count matches | `(a > 10).sum()` |
| Replace matching values | `a[a < 0] = 0` |
| Choose between two values | `np.where(a > 10, "high", "low")` |
| Pick positions | `a[[0, 3, 3]]` |
| Sorted copy / sorting positions | `np.sort(a)`, `np.argsort(a)` |
| Position of max / min | `a.argmax()`, `a.argmin()` |

## 2.11 Exercises

**1.** Using `temp` from 2.3, take Mumbai's temperatures for the last 7 days of the year, and print them in reverse order (31 December first).

```python
# Solution
mumbai_last_week = temp[1, -7:]
print(mumbai_last_week[::-1])
```

**2.** Make `x = np.arange(10)`, then `y = x[3:7]`. Set every value of `y` to 0. What does `x` look like now? Repeat with `.copy()` and check both cases with `np.shares_memory`.

```python
# Solution
x = np.arange(10)
y = x[3:7]
y[:] = 0
print(x, np.shares_memory(x, y))    # x changed: y is a view

x = np.arange(10)
y = x[3:7].copy()
y[:] = 0
print(x, np.shares_memory(x, y))    # x unchanged: y is a copy
```

**3.** For Bengaluru (row 2 of `temp` and `rain`), count the days that were both cooler than 22 °C **and** had some rain (more than 0 mm). Then count the days that had no rain at all.

```python
# Solution
blr_t, blr_r = temp[2], rain[2]
print("cool and rainy:", ((blr_t < 22) & (blr_r > 0)).sum())
print("dry days:", (blr_r == 0).sum())
```

**4.** Simulate 20 order quantities with `rng.integers(1, 6, size=20)`. Replace every 5 with 4 (a promotion caps orders at 4), then use `np.where` to make a label array that says `"single"` for quantity 1 and `"multi"` otherwise.

```python
# Solution
q = rng.integers(1, 6, size=20)
q[q == 5] = 4
labels = np.where(q == 1, "single", "multi")
print(q)
print(labels)
assert q.max() <= 4
```

**5.** Using `items` and `sold` from 2.8, print the names of the two **worst**-selling items, worst first.

```python
# Solution
worst2 = np.argsort(sold)[:2]
print(items[worst2])
```
