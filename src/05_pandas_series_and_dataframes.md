# Chapter 5: pandas Series and DataFrames

NumPy arrays (Chapter 1) are fast, but they only know positions and they want every value to share one type. Real data is a table: an order has a date, a store name, a quantity and a price, and you want to refer to columns by name, not by number. pandas adds exactly that on top of NumPy: **labels** for rows and columns, and a different type for each column.

By the end of this chapter you will be able to build a Series and a DataFrame, read their index, columns and dtypes, add and rename columns, do arithmetic on whole columns, pick rows with `.loc` and `.iloc`, and explain what Copy-on-Write means for your code. It builds on Chapters 1 to 3.

```python
import numpy as np
import pandas as pd

print("pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
```

## 5.1 Why pandas?

Try to store a tiny menu in one NumPy array and you hit two problems. Mixing text and numbers turns everything into text (see 1.5), and you have to remember that "column 2 is the price".

```python
menu_arr = np.array([["Espresso", "Coffee", 120],
                     ["Muffin", "Bakery", 110]])
print(menu_arr.dtype)        # every value became a string
print(menu_arr[:, 2])        # the prices are text now, so no maths
```

The same data as a pandas **DataFrame** keeps each column in its own type and lets you ask for columns by name.

```python
menu = pd.DataFrame({
    "item": ["Espresso", "Muffin"],
    "category": ["Coffee", "Bakery"],
    "price": [120, 110],
})
print(menu)
print(menu["price"] * 2)     # real numbers, so maths just works
```

Under the hood each numeric column is still a NumPy-style array, so you keep the speed of vectorised operations (Chapter 3) and gain names, mixed types and lots of tools for messy real data.

## 5.2 Series: one labelled column

A **Series** is a one-dimensional array with a label for every value. Those labels are called the **index**. A Series also has a `name` and a `dtype`.

```python
prices = pd.Series([120, 160, 180, 90],
                   index=["Espresso", "Cappuccino", "Latte", "Masala Chai"],
                   name="price")
print(prices)
```

The output shows the index on the left, the values on the right, and the name and dtype underneath. You can pull each part out separately.

```python
print(prices.values)   # the underlying NumPy array
print(prices.index)    # the labels
print(prices.name, prices.dtype)
```

Because values have labels, you can look them up by label, just like a dictionary, and still do whole-column maths like an array.

```python
print(prices["Latte"])
print(prices * 1.05)            # a 5% price rise, label by label
print(prices[prices > 100])     # boolean filtering works as in Chapter 2
```

If you build a Series from a dictionary, the keys become the index. Leave out the index entirely and pandas numbers the rows 0, 1, 2, and so on.

```python
seats = pd.Series({"S1": 40, "S2": 24, "S3": 55, "S4": 32}, name="seats")
print(seats)
print(pd.Series([3, 1, 2]).index)   # default RangeIndex
```

## 5.3 Building DataFrames

A **DataFrame** is a table: a set of Series that share one index. Each column is a Series with its own dtype. You will build one in three common ways.

From a **dictionary of lists**, where each key becomes a column (you saw this in 5.1):

```python
stores = pd.DataFrame({
    "store_id": ["S1", "S2", "S3", "S4"],
    "city": ["Pune", "Pune", "Mumbai", "Bengaluru"],
    "seats": [40, 24, 55, 32],
})
print(stores)
```

From a **list of dictionaries**, one per row. This is the shape you often get from a web API or a JSON file. Missing keys become missing values.

```python
rows = [
    {"item": "Latte", "qty": 2, "payment": "UPI"},
    {"item": "Cookie", "qty": 1},                    # no payment recorded
]
print(pd.DataFrame(rows))
```

From a **NumPy array**, adding the column names yourself:

```python
rng = np.random.default_rng(42)
daily = pd.DataFrame(rng.integers(80, 200, size=(3, 4)),
                     columns=["S1", "S2", "S3", "S4"])
print(daily)
```

Each row of `daily` is a day and each column a store's number of orders. In real work you will mostly load tables from files instead (Chapter 6), but these constructors are perfect for small examples and tests.

## 5.4 Index, columns and dtypes

A DataFrame has two sets of labels: the **index** (row labels) and the **columns** (column labels). It also has a `shape`, like a NumPy array.

```python
print(stores.index)
print(stores.columns)
print(stores.shape)
```

Often one column is a natural row label, such as a store id. `set_index` moves it into the index, and `reset_index` moves it back out as a normal column.

```python
by_id = stores.set_index("store_id")
print(by_id)
print(by_id.reset_index().columns.tolist())
```

### dtypes

Every column has its own dtype. `dtypes` lists them all at once.

```python
print(stores.dtypes)
```

Numbers become `int64` or `float64` as in NumPy. Text becomes `str`: since pandas 3, text columns get a dedicated string dtype instead of the general `object` dtype that older tutorials show. It only holds strings (or missing values), which makes it safer and faster.

Missing values need care. NumPy integers have no "missing" value, so one gap turns a whole integer column into floats. pandas offers **nullable** dtypes, written with a capital letter, such as `Int64`, that keep whole numbers and mark gaps as `<NA>`.

```python
ages = [25, None, 44]
print(pd.Series(ages).dtype)                  # float64: 25 became 25.0
print(pd.Series(ages, dtype="Int64"))         # stays whole, gap shown as <NA>
```

Use `astype` to convert a column, just like `astype` on an array (see 1.5).

```python
converted = stores.astype({"seats": "float64"})
print(converted.dtypes)
```

## 5.5 Adding, removing and renaming columns

Assigning to a new column name adds a column. The right-hand side can be a single value (repeated for every row), a list of the right length, or a calculation on other columns.

```python
stores2 = stores.copy()
stores2["country"] = "India"                       # one value for every row
stores2["opened"] = [2018, 2021, 2019, 2022]       # a list, one per row
stores2["seats_per_10"] = stores2["seats"] / 10    # computed
print(stores2)
```

`drop(columns=...)` returns a table without the named columns. `rename(columns={old: new})` returns one with new names. Neither changes the original; you reassign the result, which is the pandas 3 style (no `inplace=True`).

```python
stores2 = stores2.drop(columns=["country", "seats_per_10"])
stores2 = stores2.rename(columns={"seats": "n_seats", "opened": "year_opened"})
print(stores2.columns.tolist())
```

To pick a subset of columns in a chosen order, pass a list of names. Note the double brackets: the outer pair means "select", the inner pair is the Python list.

```python
print(stores2[["city", "store_id"]])
```

## 5.6 Column arithmetic and index alignment

Arithmetic between columns works element by element, exactly like NumPy arrays (Chapter 3). Here we compute the value of each order line.

```python
lines = pd.DataFrame({
    "item": ["Latte", "Cookie", "Sandwich"],
    "qty": [2, 3, 1],
    "unit_price": [180, 70, 220],
})
lines["line_total"] = lines["qty"] * lines["unit_price"]
print(lines)
print("Total:", lines["line_total"].sum())
```

There is one important difference from NumPy. When you combine two Series, pandas matches values **by label, not by position**. This is called **index alignment**. Labels that appear in only one of them give `NaN` (missing).

```python
jan = pd.Series({"S1": 1200, "S2": 800, "S3": 1500})
feb = pd.Series({"S3": 1600, "S1": 1100, "S4": 900})   # different order, S4 is new
print(jan + feb)
```

S1 and S3 were added correctly even though they sit in different positions. S2 has no February value and S4 has no January value, so both become `NaN`. If you would rather treat a missing value as zero, use the method form with `fill_value`.

```python
print(jan.add(feb, fill_value=0))
```

Alignment protects you from silently adding the wrong rows together, which is a classic bug with plain arrays. When you see unexpected `NaN` after arithmetic, the first thing to check is whether the two indexes match.

## 5.7 Selecting rows with .loc and .iloc

`df["col"]` picks columns. To pick rows (and columns at the same time) pandas has two tools:

- `.loc[rows, cols]` selects **by label**;
- `.iloc[rows, cols]` selects **by integer position**, like NumPy indexing (Chapter 2).

```python
by_id = stores.set_index("store_id")
print(by_id.loc["S3"])                    # one row, by label -> a Series
print(by_id.loc["S3", "seats"])           # one value
print(by_id.loc[["S1", "S4"], ["city"]])  # lists of labels -> a DataFrame
```

```python
print(by_id.iloc[0])          # first row
print(by_id.iloc[-1, 1])      # last row, second column
print(by_id.iloc[1:3])        # rows at positions 1 and 2
```

One difference trips people up: a label slice with `.loc` **includes** the end label, while a position slice with `.iloc` excludes the end, as in Python.

```python
print(by_id.loc["S1":"S3"].index.tolist())   # stops AT S3 (included)
print(by_id.iloc[0:2].index.tolist())        # S3 sits at position 2, which is excluded
```

Both slices "end at S3", but only `.loc` returns it.

`.loc` also accepts a boolean Series, which is how you filter rows by a condition. You will use this all the time in Chapter 6.

```python
print(by_id.loc[by_id["seats"] > 35, "city"])
```

## 5.8 Copy-on-Write and getting back to NumPy

### Copy-on-Write

When you select part of a DataFrame, is the result a copy or a view into the original (see Chapter 2 for views in NumPy)? In pandas 3 the answer is simple: it **behaves like a copy**, always. This rule is called **Copy-on-Write**: pandas shares memory while it can, and quietly makes a real copy the moment one of them is changed.

```python
small = stores[stores["seats"] > 30]
small.loc[small.index[0], "seats"] = 999      # change the subset
print(small["seats"].tolist())
print(stores["seats"].tolist())              # the original is untouched
```

The flip side is that **chained assignment** such as `stores["seats"][0] = 999` never changes `stores`, because `stores["seats"]` is treated as a separate object (pandas also warns you). To change a value in a table, do it in one step with `.loc`:

```python
stores3 = stores.copy()
stores3.loc[0, "seats"] = 45                 # row label 0, column "seats"
print(stores3.loc[0])
```

### to_numpy

Some tools (and some of your own code) want a plain NumPy array. `to_numpy()` gives you one. For numeric columns you get a numeric array; mixing text and numbers gives an `object` array.

```python
print(lines[["qty", "unit_price"]].to_numpy())
print(lines.to_numpy().dtype)
```

## 5.9 Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| `df["a", "b"]` to select two columns | `KeyError`: pandas looks for one column named `("a", "b")` | Use a list: `df[["a", "b"]]` |
| Chained assignment `df["a"][0] = 1` | `df` is not changed under Copy-on-Write | `df.loc[0, "a"] = 1` |
| Forgetting to reassign after `drop` or `rename` | The original table is unchanged | `df = df.drop(columns=["a"])` |
| Adding two Series with different indexes | Unmatched labels become `NaN` | Check the indexes, or use `.add(other, fill_value=0)` |
| Expecting `.loc["S1":"S3"]` to stop before S3 | The end label is included | Use `.iloc` for position-based, end-exclusive slices |
| Integer column with gaps turns into floats | `25` shows as `25.0` | Use the nullable `"Int64"` dtype |

## 5.10 Cheat sheet

| Task | Code |
|---|---|
| Series with labels | `pd.Series([1, 2], index=["a", "b"], name="x")` |
| DataFrame from a dict of lists | `pd.DataFrame({"a": [1, 2], "b": [3, 4]})` |
| DataFrame from rows | `pd.DataFrame([{"a": 1}, {"a": 2}])` |
| DataFrame from an array | `pd.DataFrame(arr, columns=["x", "y"])` |
| Labels, size, types | `df.index`, `df.columns`, `df.shape`, `df.dtypes` |
| Column to index and back | `df.set_index("id")`, `df.reset_index()` |
| Nullable integers | `pd.Series(vals, dtype="Int64")` |
| Add a column | `df["total"] = df["qty"] * df["price"]` |
| Drop / rename columns | `df.drop(columns=["a"])`, `df.rename(columns={"a": "b"})` |
| Add with missing as zero | `s1.add(s2, fill_value=0)` |
| Select by label | `df.loc["S1", "city"]` |
| Select by position | `df.iloc[0, 1]` |
| Set one value | `df.loc[row, "col"] = value` |
| To NumPy | `df.to_numpy()` |

## 5.11 Exercises

**1.** Build a Series `temps` of daily temperatures `[23.4, 19.7, 23.7]` with the index `["Mon", "Tue", "Wed"]` and the name `"temp_c"`. Print the Tuesday value and the days warmer than 20 degrees.

```python
# Solution
temps = pd.Series([23.4, 19.7, 23.7], index=["Mon", "Tue", "Wed"], name="temp_c")
print(temps["Tue"])
print(temps[temps > 20])
```

**2.** Create a DataFrame of three menu items (`Espresso` 120, `Croissant` 140, `Cold Brew` 200) with columns `item` and `price`. Add a column `price_with_tax` that is 5% higher, then rename `price` to `base_price`.

```python
# Solution
m = pd.DataFrame({"item": ["Espresso", "Croissant", "Cold Brew"], "price": [120, 140, 200]})
m["price_with_tax"] = m["price"] * 1.05
m = m.rename(columns={"price": "base_price"})
print(m)
```

**3.** Two stores report sales for different weeks: `a = pd.Series({"w1": 10, "w2": 12})` and `b = pd.Series({"w2": 5, "w3": 7})`. Predict `a + b`, then compute a total in which a missing week counts as zero.

```python
# Solution
a = pd.Series({"w1": 10, "w2": 12})
b = pd.Series({"w2": 5, "w3": 7})
print(a + b)                      # w1 and w3 are NaN, w2 is 17
total = a.add(b, fill_value=0)
print(total)
assert total.tolist() == [10.0, 17.0, 7.0]
```

**4.** Using `stores` from this chapter, set `store_id` as the index. Then (a) get the city of S4 with `.loc`, (b) get the same value with `.iloc`, and (c) set the seats of S2 to 30 without chained assignment.

```python
# Solution
s = stores.set_index("store_id")
print(s.loc["S4", "city"])
print(s.iloc[3, 0])
s.loc["S2", "seats"] = 30
print(s)
```
