# Chapter 6: Loading, selecting and filtering data with pandas

Chapter 5 built small DataFrames by hand. Real data arrives in files. In this chapter you will load a whole year of orders from Brew & Byte, our fictional four-store coffee chain, and learn the everyday moves of data work: take a first look, pick the columns you need, keep only the rows that matter, sort, and add calculated columns.

By the end you will be able to answer questions like "which Mumbai orders paid in cash for more than 300 rupees?" in one or two lines. It builds on Chapter 5 (Series, DataFrames, `.loc`) and on boolean masks from Chapter 2.

```python
import os
import tempfile

import numpy as np
import pandas as pd

print("pandas version:", pd.__version__)
```

## 6.1 Reading a CSV file

`pd.read_csv` reads a comma-separated file into a DataFrame. The column names come from the first line, and pandas guesses a dtype for every column.

```python
orders = pd.read_csv("data/orders.csv")
print(orders.shape)
print(orders.dtypes)
```

The guess is good, but not perfect: `timestamp` came in as plain text (`str`). A few options fix this at load time, which is better than fixing it later.

| Option | What it does |
|---|---|
| `parse_dates=["col"]` | turns those columns into real dates and times (Chapter 9) |
| `dtype={"col": "str"}` | forces a type instead of letting pandas guess |
| `usecols=[...]` | loads only some columns, which saves memory |
| `nrows=1000` | reads only the first rows, handy for a quick peek at a huge file |

```python
orders = pd.read_csv("data/orders.csv", parse_dates=["timestamp"])
print(orders["timestamp"].dtype)

peek = pd.read_csv("data/orders.csv", usecols=["order_id", "item", "qty"], nrows=5)
print(peek)
```

Forcing a dtype matters for codes that look like numbers. A PIN code such as `"040001"` would lose its leading zero if read as an integer. Our `order_id` is a true number, but it is an identifier, not a quantity, so you could keep it as text:

```python
ids = pd.read_csv("data/orders.csv", usecols=["order_id"], dtype={"order_id": "str"}, nrows=3)
print(ids["order_id"].tolist(), ids["order_id"].dtype)
```

## 6.2 A first look at the data

Before any analysis, spend a minute looking. These few methods tell you most of what you need: how big the table is, what the columns are, whether values are missing and what typical values look like.

```python
orders.head()       # first 5 rows (tail() gives the last 5)
```

```python
orders.info()
```

`info()` lists every column with its count of non-missing values and its dtype. Here `customer_id` has far fewer non-null values than there are rows: many customers order without a loyalty id. Chapter 7 deals with missing values properly.

`describe()` summarises the numeric columns: count, mean, spread and quartiles.

```python
orders[["qty", "unit_price"]].describe().round(1)
```

Every line has a quantity of 1 or 2, and unit prices range from 70 to 220 rupees, which matches the menu.

For text columns, `value_counts()` counts each distinct value (most common first) and `nunique()` counts how many distinct values there are.

```python
print(orders["store_id"].value_counts())
print("distinct items:", orders["item"].nunique())
```

S3 (the Mumbai store) has the most order lines and S2 the fewest. Pass `normalize=True` to get shares instead of counts:

```python
print(orders["payment"].value_counts(normalize=True).round(3))
```

About 60% of order lines are paid with UPI.

## 6.3 Selecting columns

One column with single brackets gives a Series. A list of columns gives a DataFrame (see 5.5).

```python
items = orders["item"]
print(type(items).__name__, items.shape)

slim = orders[["timestamp", "store_id", "item", "qty"]]
print(type(slim).__name__, slim.shape)
```

When you have many columns, you can choose them by dtype or by a pattern in the name.

```python
print(orders.select_dtypes("number").columns.tolist())
print(orders.filter(like="_id").columns.tolist())
```

## 6.4 Filtering rows with conditions

Filtering works like boolean masks in NumPy (Chapter 2). A comparison on a column gives a Series of `True`/`False`, one per row. Put that inside `[]` (or `.loc[]`) to keep the `True` rows.

```python
is_mumbai = orders["store_id"] == "S3"
print(is_mumbai.head(3))
print("Mumbai lines:", is_mumbai.sum())    # True counts as 1
```

```python
mumbai = orders[is_mumbai]
mumbai.head(3)
```

To combine conditions, use `&` (and), `|` (or) and `~` (not), and wrap **each condition in brackets**. Python's `and` and `or` do not work on whole columns.

```python
big_cash = orders[(orders["payment"] == "Cash") & (orders["qty"] == 2)]
print(len(big_cash))

not_coffee = orders[~(orders["category"] == "Coffee")]
print(not_coffee["category"].unique())
```

To filter rows and choose columns in one step, use `.loc[mask, columns]` (see 5.7).

```python
orders.loc[orders["unit_price"] >= 200, ["item", "unit_price"]].head(3)
```

## 6.5 isin, between and query

Long chains of `|` get hard to read. Three helpers make common filters shorter.

`isin` checks membership in a list, and `between` checks a range (both ends included by default).

```python
pune = orders[orders["store_id"].isin(["S1", "S2"])]
print(pune["store_id"].value_counts())

mid_price = orders[orders["unit_price"].between(100, 160)]
print(sorted(mid_price["item"].unique()))
```

`query` lets you write the condition as a string, with column names as variables. Use `and`, `or` and `not` inside the string. Refer to a Python variable with `@`.

```python
limit = 2
q = orders.query("store_id == 'S4' and category == 'Bakery' and qty >= @limit")
print(len(q))
q.head(3)
```

Both styles give the same rows. Use whichever reads more clearly to you; `query` is pleasant for long conditions, bracket masks are easier when the condition is built in code.

```python
same = orders[(orders["store_id"] == "S4") & (orders["category"] == "Bakery") & (orders["qty"] >= limit)]
assert same.equals(q)
```

## 6.6 Sorting and top-N

`sort_values` sorts by one or more columns. With several keys you can give a list of directions too.

```python
orders.sort_values("unit_price", ascending=False).head(3)
```

```python
by_store_price = orders.sort_values(["store_id", "unit_price"], ascending=[True, False])
by_store_price[["store_id", "item", "unit_price"]].head(4)
```

Rows inside S1 now appear from the most expensive item down. `sort_index` sorts by the row labels instead, which is handy after `value_counts`.

```python
print(orders["store_id"].value_counts().sort_index())
```

When you only want the top few, `nlargest` and `nsmallest` are shorter and faster than sorting everything.

```python
print(orders["item"].value_counts().nlargest(3))
```

Latte is the best seller by number of order lines.

## 6.7 Computed columns: assign and loc

The single most useful computed column in sales data is the value of each line: quantity times unit price. You can add it with plain assignment (see 5.5) or with `assign`, which returns a new DataFrame and so fits neatly into a chain of steps.

```python
orders = orders.assign(line_total=orders["qty"] * orders["unit_price"])
orders[["item", "qty", "unit_price", "line_total"]].head(3)
```

`assign` can also take a function of the DataFrame so far (a `lambda`). That way a later column can use one created earlier in the same call.

```python
summary = (
    orders.head(1000)
    .assign(is_large=lambda d: d["line_total"] >= 300,
            hour=lambda d: d["timestamp"].dt.hour)
)
summary[["line_total", "is_large", "hour"]].head(3)
```

To change values only in rows that match a condition, use `.loc[mask, column] = value`. This is the one-step pattern that works under Copy-on-Write (see 5.8). Here we label each line by its size.

```python
orders["size"] = "single"
orders.loc[orders["qty"] >= 2, "size"] = "multi"
print(orders["size"].value_counts())
```

Now a full question in one go: total revenue per store, largest first.

```python
print(orders.groupby("store_id")["line_total"].sum().sort_values(ascending=False))
```

S3 brings in the most revenue and S2 the least. `groupby` gets a full chapter of its own (Chapter 8); for now just read it as "for each store, sum the line totals".

## 6.8 Writing data out

`to_csv` writes a DataFrame back to a CSV file. Pass `index=False` unless the index holds something meaningful, otherwise you get an extra unnamed column when you read the file back. Here we write into a temporary folder that is deleted afterwards.

```python
top = orders.nlargest(5, "line_total")[["order_id", "item", "qty", "line_total"]]

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "top_lines.csv")
    top.to_csv(path, index=False)
    back = pd.read_csv(path)
    print(back)
    print(os.path.getsize(path), "bytes")
```

CSV is universal but it forgets dtypes: dates come back as text unless you use `parse_dates` again. For larger data you keep reusing, the **Parquet** format (`df.to_parquet(path)` and `pd.read_parquet(path)`) is smaller, faster and keeps dtypes. It needs an extra package such as `pyarrow` (`pip install pyarrow`), so we only mention it here.

## 6.9 Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| `df[df["a"] > 1 and df["b"] < 5]` | `ValueError: The truth value of a Series is ambiguous` | Use `&` and wrap each condition: `(df["a"] > 1) & (df["b"] < 5)` |
| Forgetting brackets around each condition | `&` binds before `==`, so you get an error or a wrong mask | `(df["x"] == 1) & (df["y"] == 2)` |
| Dates left as text | `.dt` fails and sorting is alphabetical | `pd.read_csv(..., parse_dates=["timestamp"])` |
| `to_csv` without `index=False` | An extra `Unnamed: 0` column appears on reload | `df.to_csv(path, index=False)` |
| Changing a filtered table and expecting the original to change | The original is untouched (Copy-on-Write) | Use `df.loc[mask, "col"] = value` on the original |
| Codes like `"040001"` read as numbers | Leading zeros vanish | `dtype={"code": "str"}` |

## 6.10 Cheat sheet

| Task | Code |
|---|---|
| Read a CSV with dates | `pd.read_csv(path, parse_dates=["timestamp"])` |
| Only some columns / rows | `pd.read_csv(path, usecols=["a", "b"], nrows=100)` |
| First look | `df.head()`, `df.info()`, `df.describe()`, `df.shape` |
| Count values | `df["col"].value_counts(normalize=True)` |
| Distinct values | `df["col"].nunique()`, `df["col"].unique()` |
| Several columns | `df[["a", "b"]]` |
| Filter rows | `df[(df["a"] > 1) & (df["b"] == "x")]` |
| In a list / in a range | `df["a"].isin([1, 2])`, `df["a"].between(10, 20)` |
| Query string | `df.query("a > @limit and b == 'x'")` |
| Sort by several keys | `df.sort_values(["a", "b"], ascending=[True, False])` |
| Top N rows | `df.nlargest(5, "col")` |
| Computed column | `df.assign(total=df["qty"] * df["price"])` |
| Conditional update | `df.loc[df["qty"] >= 2, "size"] = "multi"` |
| Write CSV | `df.to_csv(path, index=False)` |

## 6.11 Exercises

**1.** Load only the `store_id`, `item` and `payment` columns of `data/orders.csv`. How many order lines were paid in cash at store S2?

```python
# Solution
small = pd.read_csv("data/orders.csv", usecols=["store_id", "item", "payment"])
n = len(small[(small["store_id"] == "S2") & (small["payment"] == "Cash")])
print(n)
```

**2.** Using `orders`, find all lines for Cold Brew or Espresso at the Bengaluru store (S4) with a quantity of 2. Write it once with masks and once with `query`, and check they match.

```python
# Solution
mask = orders["item"].isin(["Cold Brew", "Espresso"]) & (orders["store_id"] == "S4") & (orders["qty"] == 2)
a = orders[mask]
b = orders.query("item in ['Cold Brew', 'Espresso'] and store_id == 'S4' and qty == 2")
assert a.equals(b)
print(len(a))
```

**3.** Show the five order lines with the highest `line_total` in store S1. Break ties by the most recent `timestamp` first.

```python
# Solution
s1 = orders[orders["store_id"] == "S1"]
s1.sort_values(["line_total", "timestamp"], ascending=[False, False])[["timestamp", "item", "qty", "line_total"]].head(5)
```

**4.** Add a column `daypart` to `orders` that is `"morning"` for hours before 12 and `"later"` otherwise, using `.loc`. Then show the share of each daypart.

```python
# Solution
orders["daypart"] = "later"
orders.loc[orders["timestamp"].dt.hour < 12, "daypart"] = "morning"
print(orders["daypart"].value_counts(normalize=True).round(3))
```
