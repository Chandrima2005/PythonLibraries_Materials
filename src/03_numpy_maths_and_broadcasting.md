# Chapter 3: NumPy maths, aggregation and broadcasting

Chapter 1 showed that `np.array(prices) * np.array(qty)` multiplies element by element. This chapter turns that idea into a habit. You will do maths on whole arrays at once, summarise them with totals, averages and percentiles, deal with missing values, and combine arrays of different shapes through **broadcasting**.

By the end you will be able to replace most number-crunching loops with one-line array expressions, choose the right `axis` for a summary, and standardise the columns of a real dataset. It builds on Chapter 1 (shapes and dtypes) and Chapter 2 (slicing and boolean masks).

```python
import time

import numpy as np

rng = np.random.default_rng(42)
```

## 3.1 Element-wise maths and ufuncs

Arithmetic operators on arrays work **element by element**: the first element with the first, the second with the second, and so on. A number on one side is applied to every element. No loop needed.

```python
price = np.array([120, 160, 180, 200])
qty = np.array([2, 1, 3, 1])

print(price * qty)          # line totals
print(price * 1.18)         # add 18% tax to every price
print(price // 100, price % 100)   # whole hundreds and remainder
```

Behind the operators sit **ufuncs** ("universal functions"): fast compiled functions that run on every element. `+` is really `np.add`, `*` is `np.multiply`. NumPy also has ufuncs for rounding, roots, logs, exponentials and more.

```python
x = np.array([1.0, 4.0, 9.0, 16.0])

print(np.sqrt(x))
print(np.log10(np.array([10, 100, 1000])))
print(np.round(np.array([2.345, 7.891]), 1))
print(np.abs(np.array([-3, 5, -1])))
```

Two-array ufuncs such as `np.maximum` compare element by element. They are different from `max`, which summarises a whole array (see 3.2).

```python
yesterday = np.array([40, 55, 38])
today = np.array([45, 50, 41])
print(np.maximum(yesterday, today))   # the better day, store by store
```

## 3.2 Aggregations: one number from many

An **aggregation** reduces many values to one: a total, an average, a spread. They are the first thing you compute when you meet new data. Every aggregation is available as a function (`np.sum(a)`) and as a method (`a.sum()`).

Let's load the daily Pune temperatures from `data/weather.csv`, as in 2.3.

```python
weather = np.loadtxt("data/weather.csv", delimiter=",", skiprows=1, usecols=(2, 3))
print(weather.shape)          # 1098 rows, 2 columns: temp_c and rain_mm

pune_t = weather[:366, 0]     # Pune is the first block of 366 days
print(pune_t.min(), pune_t.max(), pune_t.mean().round(2), pune_t.std().round(2))
```

`std` is the **standard deviation**: roughly the typical distance of a value from the mean. A small std means the values sit close together.

**Percentiles** say where a value falls in the sorted data. The 50th percentile is the median, half the values are below it. The 10th and 90th percentiles give a sense of the usual range without being thrown off by a few extreme days.

```python
print(np.median(pune_t))
print(np.percentile(pune_t, [10, 50, 90]))
```

A **cumulative** sum keeps a running total. With daily rain it answers "how much rain had fallen by day N?".

```python
pune_rain = weather[:366, 1]
running = np.cumsum(pune_rain)
print(running[[30, 151, 273, 365]].round(0))   # end of Jan, May, Sep, Dec
```

The running total barely moves until the end of May, then climbs fast through the monsoon months, and flattens again after September.

## 3.3 The axis argument

On a 2-D array an aggregation can work in three ways: over everything, down the rows, or across the columns. You choose with `axis`.

- `axis=0` **collapses the rows**: you get one result per column.
- `axis=1` **collapses the columns**: you get one result per row.
- no axis: one result for the whole array.

A good way to remember it: the axis you name is the one that **disappears** from the shape.

```python
sales = np.array([[30, 45, 12, 20],      # store S1: Espresso, Latte, Chai, Cookie
                  [25, 50, 18, 15],      # store S2
                  [40, 60, 10, 30]])     # store S3
print(sales.shape)

print(sales.sum())            # everything: one number
print(sales.sum(axis=0))      # per item (column), shape (4,)
print(sales.sum(axis=1))      # per store (row), shape (3,)
```

Here is the same idea drawn out:

```
             Espresso Latte Chai Cookie      sum(axis=1)
S1              30     45    12    20    ->     107
S2              25     50    18    15    ->     108
S3              40     60    10    30    ->     140
                 |      |     |     |
sum(axis=0)     95    155    40    65
```

The real weather table works the same way. Reshaped as cities x days (as in 2.3), `axis=1` gives one value per city.

```python
temp = weather[:, 0].reshape(3, 366)
print(temp.mean(axis=1).round(2))        # Pune, Mumbai, Bengaluru
print(temp.max(axis=0)[:5])              # hottest city's value on each of the first 5 days
```

`argmax` takes an axis too, so `temp.argmax(axis=0)` tells you **which** city was hottest each day (see 2.8).

```python
hottest_city = temp.argmax(axis=0)
print(np.bincount(hottest_city))    # how many days each city (0, 1, 2) was the hottest
```

`np.bincount` counts how often each whole number 0, 1, 2, ... appears, which is a quick way to tally category codes.

## 3.4 Missing values: NaN

Real data has gaps. In a float array a missing value is stored as `np.nan` ("not a number"). NaN is contagious: any maths that touches it gives NaN, so one gap ruins a plain `sum` or `mean`.

```python
readings = np.array([22.5, 23.1, np.nan, 24.0, 22.8])
print(readings.mean())
print(readings.sum())
```

Every common aggregation has a `nan` twin that skips missing values: `np.nansum`, `np.nanmean`, `np.nanmin`, `np.nanmax`, `np.nanstd`, `np.nanpercentile`.

```python
print(np.nanmean(readings).round(2))
print(np.nanmax(readings))
print(np.isnan(readings).sum(), "missing")
```

NaN is not even equal to itself, so `readings == np.nan` is always `False`. Always test with `np.isnan`. To fill the gaps instead of skipping them, combine `np.isnan` with mask assignment (see 2.6):

```python
print(readings == np.nan)                  # never finds anything

filled = readings.copy()
filled[np.isnan(filled)] = np.nanmean(readings)
print(filled)
```

pandas handles missing values even more conveniently, and Chapter 7 covers cleaning them in depth.

## 3.5 Broadcasting

So far we have combined arrays of the **same** shape, or an array with a single number. **Broadcasting** is the rule that lets NumPy combine arrays of **different** shapes, as long as they are compatible. You already used it: `price * 1.18` broadcasts the single number to every element.

NumPy compares the shapes **from the right**, one axis at a time. Two sizes are compatible when they are **equal**, or when **one of them is 1**. A missing axis on the left counts as 1. Size-1 axes are then stretched (without copying data) to match the other array.

Take item prices, one per column of `sales` from 3.3:

```python
item_price = np.array([120, 180, 90, 70])   # Espresso, Latte, Chai, Cookie
revenue = sales * item_price                # (3, 4) * (4,)
print(revenue)
```

Here is what happened to the shapes:

```
sales        (3, 4)
item_price      (4,)   -> treated as (1, 4) -> stretched to (3, 4)
result       (3, 4)

   30 45 12 20       120 180 90 70
   25 50 18 15   *   120 180 90 70     (the same row, reused 3 times)
   40 60 10 30       120 180 90 70
```

To work per **row** instead, the small array must be a column of shape `(3, 1)`. Say each store has a different discount:

```python
discount = np.array([0.0, 0.10, 0.05])           # S1, S2, S3
net = revenue * (1 - discount[:, np.newaxis])    # (3, 4) * (3, 1)
print(net)
```

```
revenue      (3, 4)
discount     (3, 1)  -> column stretched across 4 items -> (3, 4)
```

If the shapes do not fit, NumPy refuses rather than guessing. A `(3,)` array next to `(3, 4)` lines up 3 against 4 on the right, which fails:

```python
try:
    revenue * discount            # (3, 4) * (3,)
except ValueError as e:
    print("ValueError:", e)
```

The fix is the one above: add an axis with `[:, np.newaxis]` (see 1.6). An aggregation with `keepdims=True` keeps the collapsed axis as size 1, so its result broadcasts straight back against the original. This is how you compare each value with its own row's mean:

```python
city_mean = temp.mean(axis=1, keepdims=True)    # shape (3, 1)
diff = temp - city_mean                         # (3, 366) - (3, 1)
print(city_mean.shape, diff.shape)
print(diff[:, :4].round(1))                      # first 4 days vs each city's own average
```

Broadcasting also builds whole tables from two 1-D arrays. A column times a row gives every combination:

```python
cups = np.arange(1, 5)[:, np.newaxis]     # 1..4 cups, shape (4, 1)
prices = np.array([120, 160, 180])        # shape (3,)
print(cups * prices)                      # (4, 3) price table
```

## 3.6 Vectorising a loop

**Vectorising** means replacing a Python loop over elements with one array expression. The code gets shorter and usually much faster. The recipe: turn each `if` into a mask or `np.where` (see 2.6), and each running variable into an aggregation.

Let's take every order line from `data/orders.csv` (53,000 rows). The rule: a line with 3 or more items gets a 10% discount. Here is the loop version first.

```python
lines = np.loadtxt("data/orders.csv", delimiter=",", skiprows=1, usecols=(6, 7))
qty, unit_price = lines[:, 0], lines[:, 1]
print(lines.shape)

def bill_loop(qty, unit_price):
    totals = []
    for q, p in zip(qty, unit_price):
        t = q * p
        if q >= 3:
            t = t * 0.9
        totals.append(t)
    return np.array(totals)
```

And the vectorised version, which is a single line:

```python
def bill_vec(qty, unit_price):
    return qty * unit_price * np.where(qty >= 3, 0.9, 1.0)
```

Now check that both give the same answer and time them.

```python
t0 = time.perf_counter(); a = bill_loop(qty, unit_price); t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); b = bill_vec(qty, unit_price);  t_vec = time.perf_counter() - t0

assert np.allclose(a, b)
print(f"total revenue: {b.sum():,.0f}")
print(f"loop: {t_loop*1000:.1f} ms   vectorised: {t_vec*1000:.2f} ms   speed-up: {t_loop/t_vec:.0f}x")
```

The `assert` proves the two versions agree, so the speed-up is free. Always keep a check like this when you rewrite a loop: a fast wrong answer is no use. Exact timings depend on your machine.

## 3.7 Comparisons, any, all and isclose

Comparison operators give boolean arrays (see 2.5). Two aggregations summarise them: `any` asks "is at least one `True`?" and `all` asks "are they all `True`?". Both accept `axis`.

```python
print((pune_t > 32).any())          # was Pune ever above 32 °C?
print((pune_t > 10).all())          # was every day above 10 °C?
print((temp > 30).any(axis=1))      # per city: any day above 30 °C?
```

Comparing floats with `==` is risky, because decimals are stored in binary and tiny rounding errors creep in.

```python
print(0.1 + 0.2 == 0.3)
print(0.1 + 0.2)
```

`np.isclose` compares with a small tolerance, element by element. `np.allclose` is the "all" version that returns one `True` or `False`, which is what you want inside an `assert`.

```python
a = np.array([0.1 + 0.2, 1.0, 2.5])
b = np.array([0.3, 1.0, 2.5000001])
print(a == b)
print(np.isclose(a, b))
print(np.allclose(a, b))
```

For integer arrays exact equality is fine, and `np.array_equal(a, b)` checks both shape and values in one go.

## 3.8 Worked example: standardising columns with z-scores

Temperature (around 25 °C) and rainfall (0 to dozens of mm) live on very different scales, so it is hard to say whether a day was "more unusual" in heat or in rain. **Standardising** puts every column on the same scale: subtract the column mean, then divide by the column standard deviation. The result is a **z-score**: 0 means average, +2 means two standard deviations above average. Machine learning models in the next stage of the pathway often expect inputs scaled like this.

With `axis=0` and broadcasting this takes two lines, for any number of columns.

```python
X = weather                                 # (1098, 2): temp_c, rain_mm
mu = X.mean(axis=0)                         # (2,)  one mean per column
sigma = X.std(axis=0)                       # (2,)  one std per column
Z = (X - mu) / sigma                        # (1098, 2) - (2,) -> broadcast down the rows

print(mu.round(2), sigma.round(2))
print(Z[:3].round(2))
```

Each column of `Z` should now have mean 0 and standard deviation 1. Floating-point arithmetic leaves the mean at something like `1e-16` rather than exactly 0, which is why we check with `np.allclose`:

```python
assert np.allclose(Z.mean(axis=0), 0)
assert np.allclose(Z.std(axis=0), 1)
print(Z.mean(axis=0), Z.std(axis=0))
```

Now the two columns can be compared. Which day was the most extreme for rain, and how unusual was it?

```python
wettest = Z[:, 1].argmax()
print("row", wettest, "rain z-score:", Z[wettest, 1].round(1), "->", X[wettest, 1], "mm")
print("days with |z| > 3 for rain:", (np.abs(Z[:, 1]) > 3).sum())
```

A z-score above 3 is a common rule of thumb for "unusual". Note that NumPy's `std` divides by `n` by default; pandas divides by `n - 1`. Pass `ddof=1` to NumPy if you need the pandas result (Chapter 5).

## 3.9 Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Mixing up `axis=0` and `axis=1` | You get one value per row when you wanted one per column | Remember: the named axis disappears. `axis=0` gives one result per column |
| `mean` on data with NaN | Result is `nan` | Use `np.nanmean` (and the other `nan` functions) |
| Testing `a == np.nan` | Always `False`, even for missing values | `np.isnan(a)` |
| Subtracting row results without `keepdims` | `ValueError: operands could not be broadcast` for `(3, 366) - (3,)` | `a.mean(axis=1, keepdims=True)` or `[:, np.newaxis]` |
| Comparing floats with `==` | `0.1 + 0.2 == 0.3` is `False` | `np.isclose`, or `np.allclose` in an assert |
| Using `max(a, b)` for two arrays | Error or wrong result | `np.maximum(a, b)` for element-wise; `a.max()` for a summary |

## 3.10 Cheat sheet

| Task | Code |
|---|---|
| Element-wise maths | `a * b`, `a + 1`, `np.sqrt(a)`, `np.log(a)` |
| Element-wise max of two arrays | `np.maximum(a, b)` |
| Total, mean, spread | `a.sum()`, `a.mean()`, `a.std()` |
| Median and percentiles | `np.median(a)`, `np.percentile(a, [10, 90])` |
| Running total | `np.cumsum(a)` |
| One result per column / per row | `m.sum(axis=0)`, `m.sum(axis=1)` |
| Keep the axis for broadcasting | `m.mean(axis=1, keepdims=True)` |
| Ignore NaN | `np.nanmean(a)`, `np.nansum(a)` |
| Find NaN | `np.isnan(a)` |
| Make a column for broadcasting | `v[:, np.newaxis]` |
| Any / all true | `(a > 0).any()`, `(a > 0).all(axis=1)` |
| Float comparison | `np.isclose(a, b)`, `np.allclose(a, b)` |
| Count category codes | `np.bincount(codes)` |
| Z-scores per column | `(X - X.mean(axis=0)) / X.std(axis=0)` |

## 3.11 Exercises

**1.** Using `sales` and `item_price` from 3.3 and 3.5, compute the total revenue of each store and the name of the store with the highest revenue.

```python
# Solution
stores = np.array(["S1", "S2", "S3"])
store_revenue = (sales * item_price).sum(axis=1)
print(store_revenue)
print(stores[store_revenue.argmax()])
```

**2.** Using `temp` (cities x days) from 3.3, compute each city's temperature range (max minus min) in one expression, and print the 25th and 75th percentile for each city.

```python
# Solution
print((temp.max(axis=1) - temp.min(axis=1)).round(1))
print(np.percentile(temp, [25, 75], axis=1).round(1))   # rows: 25th, 75th; columns: cities
```

**3.** Rewrite this loop as one vectorised line and prove with `assert` that the results match: for each daily rainfall in `pune_rain`, the "umbrella score" is `0` if rain is 0, `1` if rain is up to 10 mm, and `2` above that.

```python
# Solution
def umbrella_loop(r):
    out = []
    for x in r:
        out.append(0 if x == 0 else (1 if x <= 10 else 2))
    return np.array(out)

score = np.where(pune_rain == 0, 0, np.where(pune_rain <= 10, 1, 2))
assert np.array_equal(score, umbrella_loop(pune_rain))
print(np.bincount(score))     # how many days got score 0, 1, 2
```

**4.** Simulate a 4 x 5 grid of shop-floor sensor readings with `rng.normal(25, 2, size=(4, 5))`, then set two readings to `np.nan`. Compute the mean of each row while ignoring the gaps, and subtract it from every value in its row.

```python
# Solution
g = rng.normal(25, 2, size=(4, 5))
g[0, 1] = np.nan
g[3, 4] = np.nan
row_mean = np.nanmean(g, axis=1, keepdims=True)
print(row_mean.ravel().round(2))
print((g - row_mean).round(2))     # NaN stays NaN; everything else is centred
```

**5.** For a 1,000 x 3 array `M = rng.integers(0, 100, size=(1000, 3))`, scale each column to the range 0 to 1 ("min-max scaling": subtract the column minimum, divide by the column range). Check that every column now has minimum 0 and maximum 1.

```python
# Solution
M = rng.integers(0, 100, size=(1000, 3))
lo, hi = M.min(axis=0), M.max(axis=0)
scaled = (M - lo) / (hi - lo)
assert np.allclose(scaled.min(axis=0), 0) and np.allclose(scaled.max(axis=0), 1)
print(scaled[:3].round(3))
```
