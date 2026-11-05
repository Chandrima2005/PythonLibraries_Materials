# Chapter 7: Cleaning data with pandas

Real data is never as tidy as the files you loaded in Chapter 6. Somebody types "latte" with spaces around it, a till exports prices as "Rs. 180", a store id is written in lower case, a row is saved twice. If you analyse such data directly, your counts and totals will be quietly wrong.

In this chapter you will clean `orders_messy.csv`, a deliberately messy export of 1,525 Brew & Byte order lines, plus the loyalty-card table `customers.csv`. By the end you will have a cleaning checklist, a toolkit for each common problem, and a reusable `clean_orders()` function that you can trust because it checks its own result. It builds on Chapters 5 and 6.

```python
import numpy as np
import pandas as pd

raw = pd.read_csv("data/orders_messy.csv")
customers = pd.read_csv("data/customers.csv")
menu = pd.read_csv("data/menu.csv")

print(raw.shape, customers.shape, menu.shape)
```

## 7.1 A cleaning checklist

Cleaning goes faster when you follow the same steps every time. This is the order used in this chapter:

| Step | Question to ask | Main tools |
|---|---|---|
| 1. Look | What does the data look like? Which dtypes are wrong? | `head`, `info`, `value_counts` |
| 2. Missing values | Where are the gaps, and what do they mean? | `isna`, `dropna`, `fillna` |
| 3. Duplicates | Is any row recorded twice? | `duplicated`, `drop_duplicates` |
| 4. Types | Are numbers stored as numbers? | `str.replace`, `pd.to_numeric` |
| 5. Text | Are labels spelled consistently? | `str.strip`, `str.title`, `str.upper` |
| 6. Dates | Are dates real dates, read the right way round? | `pd.to_datetime(format=...)` |
| 7. Outliers | Are some values impossible or suspicious? | quantiles, the IQR rule |
| 8. Validate | Does the result meet every rule we expect? | `assert` |

Always keep the raw data unchanged and build a cleaned copy. Then you can rerun the cleaning from scratch at any time.

Step 1 is a look at the data. Two warning signs appear straight away in `info()`: `qty` is a float (whole numbers with gaps become floats, see 5.4) and `unit_price` is text, not a number.

```python
raw.info()
```

```python
print(raw["store_id"].value_counts())
print(raw["item"].value_counts().tail(3))
```

The counts show a lower-case `s3` and an item `'  latte '` with stray spaces. Each would be counted as a separate store or product.

## 7.2 Missing values

`isna()` marks each missing value with `True`. Summing it counts the gaps per column, and taking the mean gives the share.

```python
missing = pd.DataFrame({"count": raw.isna().sum(), "share": raw.isna().mean().round(3)})
missing[missing["count"] > 0]
```

Two columns have gaps, and they mean very different things:

- `customer_id` is missing when a guest pays without a loyalty card. That is **real information**, not an error. We keep those rows and leave the gap.
- `qty` is missing on 40 lines. Every order line sold at least one item, so the gap is a recording error.

You have three choices for a gap: drop the row (`dropna`), fill it with a sensible value (`fillna`), or leave it. Look at the column before deciding.

```python
print(raw["qty"].value_counts(dropna=False))
```

Nearly every line has a quantity of 1 or 2, and 1 is by far the most common (it is also the median). Filling the 40 gaps with 1 keeps those sales in the totals at the most likely value. Dropping them would lose 40 real sales.

```python
qty_filled = raw["qty"].fillna(1)
print("gaps before:", raw["qty"].isna().sum(), " after:", qty_filled.isna().sum())

print("rows if we dropped instead:", len(raw.dropna(subset=["qty"])))
```

`dropna(subset=[...])` only looks at the listed columns. Without `subset`, `dropna()` would also throw away every guest order because of the missing `customer_id`, which would be a serious mistake.

The `age` column of `customers` has gaps too. For a column you will average, filling with the median is a common choice, but it shrinks the spread artificially. Here we prefer to keep the gaps and store age as a nullable integer (see 5.4), so later code can decide what to do.

```python
print("missing ages:", customers["age"].isna().sum())
ages = customers["age"].astype("Int64")
print(ages.head(), "\nmedian age:", ages.median())
```

Aggregations such as `median` skip missing values automatically.

## 7.3 Duplicates

An exact duplicate is a row identical in every column, often caused by an export that ran twice. `duplicated()` marks every repeat after the first occurrence.

```python
print("exact duplicates:", raw.duplicated().sum())
raw[raw.duplicated(keep=False)].sort_values("order_id").head(4)
```

`keep=False` marks **all** copies, so you can see the pairs side by side. `drop_duplicates()` keeps the first copy of each.

```python
orders = raw.drop_duplicates()
print(len(raw), "->", len(orders))
```

Be careful with duplicates on a subset of columns. In this data one `order_id` can have several lines (a latte and a croissant in one order), so `drop_duplicates(subset=["order_id"])` would delete real sales. Only use `subset` when you know which columns should be unique.

```python
print("order ids that appear more than once:", orders["order_id"].duplicated().sum())
```

## 7.4 Fixing types

`unit_price` loaded as text because a few cells hold `"Rs. 180"` instead of `180`. `pd.to_numeric` converts text to numbers. With `errors="coerce"`, anything it cannot convert becomes `NaN` instead of raising an error, which is a quick way to find the bad cells.

```python
as_number = pd.to_numeric(orders["unit_price"], errors="coerce")
bad = orders.loc[as_number.isna(), "unit_price"]
print(len(bad), "cells failed:", bad.unique())
```

All the failures share one pattern, so we remove the prefix with `str.replace` and convert again. `regex=False` treats the pattern as plain text (the `.` would otherwise mean "any character").

```python
price = pd.to_numeric(orders["unit_price"].str.replace("Rs. ", "", regex=False))
print(price.dtype, price.isna().sum(), "missing")
```

Do not stop at "it converted". Check the values make sense. Every item has a fixed price on the menu, so compare. `map` looks up each item in a Series whose index is the item name.

```python
menu_price = menu.set_index("item")["price"]
expected = orders["item"].str.strip().str.title().map(menu_price)
wrong = price != expected
print("prices that differ from the menu:", wrong.sum())
orders.loc[wrong, ["item", "unit_price"]].head(4)
```

Fourteen of the "Rs. 180" cells are on items that do not cost 180 at all (a muffin, a cookie). Whatever system wrote them also wrote the wrong price. The menu is the trustworthy source, so the cleaning function in 7.8 takes the price from the menu for those rows.

Converting `qty` from float to a whole-number type is the last type fix. It only works once there are no gaps left.

```python
qty = orders["qty"].fillna(1).astype("int64")
print(qty.dtype, qty.unique())
```

## 7.5 Cleaning text and standardising categories

Text columns support a family of `.str` methods that work on every value at once. The ones you will use most:

| Method | Effect on `"  latte "` |
|---|---|
| `.str.strip()` | `"latte"` (removes spaces at both ends) |
| `.str.title()` | `"  Latte "` |
| `.str.lower()` / `.str.upper()` | `"  latte "` / `"  LATTE "` |

Chain them to get one consistent spelling.

```python
item_clean = orders["item"].str.strip().str.title()
print(orders["item"].nunique(), "->", item_clean.nunique(), "distinct items")
print(item_clean.value_counts().head(3))
```

The stray `'  latte '` lines have joined the real `Latte` count, which makes Latte the best seller. Store ids follow the opposite convention (upper case), so they get `upper`:

```python
store_clean = orders["store_id"].str.strip().str.upper()
print(store_clean.value_counts().sort_index())
```

The same problem appears in `customers`: a few loyalty tiers are written in lower case.

```python
print(customers["tier"].value_counts())
tier = customers["tier"].str.title()
print(tier.value_counts())
```

After standardising, it is good practice to check every value is one you expect. `isin` (see 6.5) does that in one line.

```python
assert tier.isin(["Bronze", "Silver", "Gold"]).all()
assert item_clean.isin(menu["item"]).all()
print("all categories valid")
```

## 7.6 Parsing day-first dates

The messy file stores timestamps as text like `"21/04/2024 16:49"`: day first, then month. Left to guess, a date parser may read `"03/04/2024"` as 4 March (month first) instead of 3 April. The safe way is to say exactly what the format is with `format=`.

```python
print(orders["timestamp"].head(3).tolist())
ts = pd.to_datetime(orders["timestamp"], format="%d/%m/%Y %H:%M")
print(ts.head(3))
```

The codes mean: `%d` day, `%m` month, `%Y` four-digit year, `%H` hour (24-hour clock), `%M` minute. If any value does not match the format, pandas raises an error, which is exactly what you want: you find out instead of getting silently wrong dates. A quick range check confirms everything fell inside 2024.

```python
print(ts.min(), "to", ts.max())
```

The `joined` column of `customers` is already in the unambiguous year-month-day form, so it needs no format.

```python
joined = pd.to_datetime(customers["joined"])
print(joined.dtype, joined.min().date(), joined.max().date())
```

Chapter 9 does much more with dates once they are parsed.

## 7.7 Outliers: spot them, then decide

An **outlier** is a value far from the rest. Some are errors, some are rare but real (a big office order). A simple, widely used rule is the **IQR rule**: compute the quartiles Q1 and Q3, their gap IQR = Q3 − Q1, and flag anything below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR.

```python
q1, q3 = qty.quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Q1={q1}, Q3={q3}, IQR={iqr}, keep values in [{low}, {high}]")

outliers = orders[(qty < low) | (qty > high)]
print(len(outliers), "outliers")
outliers[["order_id", "item", "qty", "unit_price"]]
```

The rule only **flags** rows; you decide what to do with them. Here every outlier has a quantity of exactly 40 on a single drink or snack, while all other lines have 1 or 2. Forty identical espressos on one order line is not believable, and we have no way to know the true number, so we drop these five lines and record the decision. With different data (say, catering orders that really are large) you might keep them, or analyse them separately.

The IQR rule works best on fairly symmetric data. For very skewed columns such as incomes, it flags many genuine values, so always look at the flagged rows before acting. Chapter 11 shows box plots, which draw exactly these IQR fences.

## 7.8 Validate, then wrap it all in a function

Cleaning code you run once in scattered cells is hard to repeat. Put each step in a small function that takes a DataFrame and returns a new one, then chain the steps with `.pipe`. `df.pipe(f, x)` simply means `f(df, x)`, but it reads top to bottom like a recipe.

```python
def fix_text(df):
    return df.assign(item=df["item"].str.strip().str.title(),
                     store_id=df["store_id"].str.strip().str.upper())

def fix_types(df, menu_price):
    price = pd.to_numeric(df["unit_price"].str.replace("Rs. ", "", regex=False))
    return df.assign(
        timestamp=pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M"),
        qty=df["qty"].fillna(1).astype("int64"),
        unit_price=price.where(price == df["item"].map(menu_price), df["item"].map(menu_price)),
    )
```

`where(cond, other)` keeps a value where `cond` is `True` and takes it from `other` elsewhere, so wrong prices are replaced by the menu price. Two more steps and the full pipeline:

```python
def drop_qty_outliers(df, max_qty=10):
    return df[df["qty"] <= max_qty]

def clean_orders(raw, menu):
    menu_price = menu.set_index("item")["price"]
    return (raw
            .drop_duplicates()
            .pipe(fix_text)
            .pipe(fix_types, menu_price)
            .pipe(drop_qty_outliers)
            .reset_index(drop=True))

clean = clean_orders(raw, menu)
print(raw.shape, "->", clean.shape)
clean.head(3)
```

`fix_text` runs before `fix_types`, so the menu lookup sees clean item names. Note that `drop_qty_outliers` uses a fixed limit of 10 rather than recomputing the IQR: once you have decided what counts as impossible, a clear fixed rule is easier to explain and to reuse on next month's file.

Now **prove** the result is clean. Each `assert` states one rule; if a future file breaks a rule, the cell stops with an error instead of letting bad data through.

```python
assert clean.duplicated().sum() == 0
assert clean["qty"].between(1, 10).all() and clean["qty"].notna().all()
assert clean["store_id"].isin(["S1", "S2", "S3", "S4"]).all()
assert clean["item"].isin(menu["item"]).all()
assert (clean["unit_price"] == clean["item"].map(menu.set_index("item")["price"])).all()
assert clean["timestamp"].dt.year.eq(2024).all()
print("all checks passed")
print(clean.dtypes)
```

All columns now have proper dtypes: dates, integers and text. From 1,525 raw lines we removed 25 duplicates and 5 impossible quantities, leaving 1,495.

## 7.9 Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| `df.dropna()` with no `subset` | Rows with harmless gaps (guest orders) disappear too | `df.dropna(subset=["qty"])` |
| `fillna(0)` for a missing quantity | Real sales count as zero and totals drop | Choose a value that makes sense (here 1), or drop the rows |
| `str.replace("Rs. ", "")` thinking of regex | With `regex=True`, `.` matches any character | Pass `regex=False` for plain text |
| Letting pandas guess day-first dates | `03/04` can become 4 March | `pd.to_datetime(s, format="%d/%m/%Y %H:%M")` |
| `drop_duplicates(subset=["order_id"])` | Deletes real lines of multi-item orders | Use exact duplicates, or a subset you know must be unique |
| Deleting every IQR outlier automatically | Genuine rare values are lost | Look at flagged rows, then decide and document |

## 7.10 Cheat sheet

| Task | Code |
|---|---|
| Count gaps per column | `df.isna().sum()` |
| Drop rows with gaps in some columns | `df.dropna(subset=["qty"])` |
| Fill gaps | `df["qty"].fillna(1)` |
| Find / drop exact duplicates | `df.duplicated().sum()`, `df.drop_duplicates()` |
| Text to number, bad cells to NaN | `pd.to_numeric(s, errors="coerce")` |
| Remove a fixed prefix | `s.str.replace("Rs. ", "", regex=False)` |
| Tidy text | `s.str.strip().str.title()` |
| Parse a known date format | `pd.to_datetime(s, format="%d/%m/%Y %H:%M")` |
| Look up values in another table | `s.map(menu.set_index("item")["price"])` |
| Keep or replace by condition | `s.where(cond, other)` |
| IQR fences | `q1, q3 = s.quantile([0.25, 0.75])` |
| Chain cleaning steps | `df.pipe(step1).pipe(step2, arg)` |
| Check a rule | `assert s.isin(allowed).all()` |

## 7.11 Exercises

**1.** Write a function `clean_customers(df)` that title-cases `tier`, parses `joined` as a date and stores `age` as `Int64`. Check that only three tiers remain.

```python
# Solution
def clean_customers(df):
    return df.assign(tier=df["tier"].str.title(),
                     joined=pd.to_datetime(df["joined"]),
                     age=df["age"].astype("Int64"))

cust = clean_customers(customers)
assert sorted(cust["tier"].unique()) == ["Bronze", "Gold", "Silver"]
print(cust.dtypes)
```

**2.** In the raw messy file, how many order lines have **both** a missing `qty` and a missing `customer_id`?

```python
# Solution
both = raw["qty"].isna() & raw["customer_id"].isna()
print(both.sum())
```

**3.** Apply the IQR rule to `unit_price` in `clean`. Are any prices flagged? Explain why or why not.

```python
# Solution
q1, q3 = clean["unit_price"].quantile([0.25, 0.75])
iqr = q3 - q1
flagged = clean[(clean["unit_price"] < q1 - 1.5 * iqr) | (clean["unit_price"] > q3 + 1.5 * iqr)]
print(q1, q3, len(flagged))
# None: every price is a menu price between 70 and 220, all well inside the fences.
```

**4.** Suppose a new export also writes some prices as `"Rs.200"` (no space). Make the price conversion robust to both forms and test it on a small Series.

```python
# Solution
test = pd.Series(["180", "Rs. 180", "Rs.200"])
fixed = pd.to_numeric(test.str.replace("Rs.", "", regex=False).str.strip())
print(fixed.tolist())
assert fixed.tolist() == [180, 180, 200]
```
