# Chapter 8: Grouping, joining and reshaping

Most business questions are about groups: revenue **per store**, orders **per hour**, average spend **per loyalty tier**. Answering them often means combining tables too, because the city of a store lives in `stores.csv` and the tier of a customer lives in `customers.csv`. And sometimes the answer is right but the table has the wrong shape for a report or a chart.

This chapter covers all three moves: grouping with `groupby`, joining with `merge` and `concat`, and reshaping with `pivot_table`, `melt`, `stack` and `unstack`. It builds on Chapter 6 (loading and filtering) and uses the clean `orders.csv`; Chapter 7 showed how to get messy data to this state.

```python
import numpy as np
import pandas as pd

orders = pd.read_csv("data/orders.csv", parse_dates=["timestamp"])
orders = orders.assign(line_total=orders["qty"] * orders["unit_price"])
stores = pd.read_csv("data/stores.csv")
customers = pd.read_csv("data/customers.csv").assign(tier=lambda d: d["tier"].str.title())
menu = pd.read_csv("data/menu.csv")

print(orders.shape, stores.shape, customers.shape, menu.shape)
```

The loyalty tiers are title-cased on load, the fix from 7.5.

## 8.1 Split, apply, combine

Every grouping operation follows the same three steps:

1. **Split** the rows into groups by the value of a key column (one group per store).
2. **Apply** a calculation to each group separately (sum the line totals).
3. **Combine** the results into a new table (one row per store).

You could do it by hand with a loop and a filter (Chapter 6). It works, but it is slow and long.

```python
by_hand = {}
for store in sorted(orders["store_id"].unique()):
    by_hand[store] = int(orders.loc[orders["store_id"] == store, "line_total"].sum())
print(by_hand)
```

`groupby` does all three steps in one line, and the result is a Series indexed by the group key.

```python
revenue = orders.groupby("store_id")["line_total"].sum()
print(revenue)
assert revenue.to_dict() == by_hand
```

Read it aloud as: "group the orders by store, take the line total, sum it". S3 (Mumbai) has the highest revenue and S2 the lowest.

## 8.2 Aggregating with agg

`agg` applies one or more summary functions at once. A list of function names gives one column per function.

```python
orders.groupby("store_id")["line_total"].agg(["sum", "mean", "count"]).round(1)
```

For real reports you usually want clear column names and different columns summarised in different ways. **Named aggregation** does both: each keyword becomes an output column, set to a pair `(input column, function)`.

```python
store_summary = orders.groupby("store_id").agg(
    revenue=("line_total", "sum"),
    lines=("line_total", "size"),
    orders=("order_id", "nunique"),
    avg_qty=("qty", "mean"),
)
store_summary.round(2)
```

`size` counts rows, while `nunique` counts distinct order ids. The gap between `lines` and `orders` tells you many orders have more than one line. A derived column finishes the summary.

```python
store_summary["avg_order_value"] = (store_summary["revenue"] / store_summary["orders"]).round(1)
store_summary.sort_values("avg_order_value", ascending=False).round(2)
```

## 8.3 Several keys and the MultiIndex

Pass a list of keys to group by every combination, for example each store and category.

```python
sc = orders.groupby(["store_id", "category"])["line_total"].sum()
sc.head(6)
```

The result has a **MultiIndex**: two levels of row labels. You can select from it with `.loc` using the outer label, or a tuple for both.

```python
print(sc.loc["S2"])
print(sc.loc[("S2", "Tea")])
```

A MultiIndex is compact, but most later steps (merging, plotting, saving to CSV) are easier with ordinary columns. `reset_index()` turns the index levels back into columns (see 5.4). Alternatively, pass `as_index=False` to `groupby`.

```python
flat = sc.reset_index()
print(flat.head(3))
print(flat.columns.tolist())
```

## 8.4 transform and filter

`agg` shrinks each group to one row. Sometimes you want a group statistic **next to every original row**, for example to compute each line's share of its store's revenue. `transform` returns a result with the same index as the input, so it lines up with the rows.

```python
store_total = orders.groupby("store_id")["line_total"].transform("sum")
orders["share_of_store"] = orders["line_total"] / store_total
orders[["store_id", "line_total", "share_of_store"]].head(3)
```

The same idea at a coarser level gives each category's share of its store's revenue. First group by both keys, then divide by a `transform` over the store level.

```python
cat_rev = orders.groupby(["store_id", "category"], as_index=False)["line_total"].sum()
cat_rev["share"] = cat_rev["line_total"] / cat_rev.groupby("store_id")["line_total"].transform("sum")
cat_rev[cat_rev["store_id"] == "S1"].round(3)
```

Coffee brings in more than half of S1's revenue. The shares within each store add up to 1:

```python
print(cat_rev.groupby("store_id")["share"].sum().round(6))
```

`filter` keeps or drops **whole groups** based on a condition about the group. Here we keep only the lines of loyal customers: those with at least 70 order lines in the year. The function you pass receives one group (a small DataFrame) and must return `True` or `False`.

```python
members = orders.dropna(subset=["customer_id"])
loyal = members.groupby("customer_id").filter(lambda g: len(g) >= 70)
print(len(members), "member lines ->", len(loyal), "lines from", loyal["customer_id"].nunique(), "loyal customers")
```

The result keeps the original rows (not one row per group), so you can analyse those customers' orders further.

## 8.5 Joining tables with merge

`orders` knows the `store_id` but not the city. `stores` knows the city. `merge` joins the two tables wherever the key column matches, like a lookup in a spreadsheet or a SQL join.

```python
with_city = orders.merge(stores[["store_id", "city"]], on="store_id")
print(with_city.shape)
with_city.groupby("city")["line_total"].sum()
```

Pune has two stores (S1 and S2), so its revenue is their total combined.

### Inner and left joins

The `how=` argument decides what happens to rows without a match:

| `how=` | Keeps |
|---|---|
| `"inner"` (default) | only keys found in **both** tables |
| `"left"` | every row of the left table; unmatched ones get `NaN` |
| `"right"` / `"outer"` | every row of the right table / of both |

The difference matters. Imagine an old menu file that does not list Cookie yet:

```python
old_menu = menu[menu["item"] != "Cookie"][["item", "price"]]

inner = orders.merge(old_menu, on="item", how="inner")
left = orders.merge(old_menu, on="item", how="left")
print(len(orders), len(inner), len(left))
print("lines with no menu price:", left["price"].isna().sum())
```

The inner join silently dropped every Cookie line, so any revenue computed from it would be too low. The left join kept all rows and the `NaN` prices point at the problem. When in doubt, use a left join and check for gaps.

### indicator= and validate=

`indicator=True` adds a column `_merge` saying where each row came from. It is the quickest way to audit a join. Here we attach the loyalty tier to every order line.

```python
with_tier = orders.merge(customers[["customer_id", "tier"]], on="customer_id",
                         how="left", indicator=True)
print(with_tier["_merge"].value_counts())
```

The `left_only` rows are the guest orders with no `customer_id` (see 7.2); every member order found its customer. 

`validate=` checks the relationship you expect between the keys. Each order line should match **at most one** customer, so the join is `"many_to_one"`. If `customers` ever contained a duplicated id, the merge would raise an error instead of silently doubling those order lines.

```python
checked = orders.merge(customers[["customer_id", "tier"]], on="customer_id",
                       how="left", validate="many_to_one")
assert len(checked) == len(orders)   # a left many-to-one join never adds rows
checked.dropna(subset=["tier"]).groupby("tier")["line_total"].mean().round(1)
```

If both tables have a column with the same name that is not the key, pandas adds suffixes `_x` and `_y`. `orders` and `menu` both have `category`, so choose the columns you need or set `suffixes=`.

```python
both = orders.merge(menu, on="item", suffixes=("_order", "_menu"))
print([c for c in both.columns if c.startswith("category")])
assert (both["category_order"] == both["category_menu"]).all()
```

## 8.6 Stacking tables with concat

`merge` puts tables side by side by matching keys. `pd.concat` stacks tables **on top of each other**, for example monthly exports into one year, or a newly opened store appended to `stores`.

```python
new_store = pd.DataFrame({"store_id": ["S5"], "city": ["Hyderabad"], "opened": [2025], "seats": [36]})
all_stores = pd.concat([stores, new_store], ignore_index=True)
all_stores
```

`ignore_index=True` builds a fresh 0, 1, 2, ... index instead of keeping the old labels, which would repeat. The tables should have the same columns; missing columns are filled with `NaN`.

```python
jan = orders[orders["timestamp"].dt.month == 1]
feb = orders[orders["timestamp"].dt.month == 2]
jan_feb = pd.concat([jan, feb], ignore_index=True)
print(len(jan), "+", len(feb), "=", len(jan_feb))
```

## 8.7 Pivot tables and crosstab

A **pivot table** is a grouped summary laid out as a grid: one key down the rows, another across the columns. It is the same calculation as grouping by two keys (8.3), shaped for reading.

```python
grid = orders.pivot_table(index="store_id", columns="category",
                          values="line_total", aggfunc="sum")
grid
```

Coffee is the largest category in every store. `margins=True` adds row and column totals, labelled `All`.

```python
orders.pivot_table(index="store_id", columns="payment", values="line_total",
                   aggfunc="sum", margins=True)
```

`pd.crosstab` is a shortcut for counting how often each combination occurs. With `normalize="index"`, each row shows shares that add up to 1.

```python
pd.crosstab(orders["store_id"], orders["payment"], normalize="index").round(3)
```

Every store has a similar payment mix, with UPI used for roughly 60% of lines.

## 8.8 Reshaping: melt, stack and unstack

A table can be **wide** (one column per store, as in `grid`) or **long** (one row per store-and-category pair with a single value column). Wide is easier for people to read. Long is what grouping, merging and plotting libraries such as Seaborn (Chapter 11) prefer.

`melt` turns wide into long. `id_vars` are the columns to keep as they are; every other column becomes a row.

```python
wide = grid.reset_index()
long = wide.melt(id_vars="store_id", var_name="category", value_name="revenue")
print(wide.shape, "->", long.shape)
long.head(5)
```

Going back from long to wide is `pivot` (no aggregation needed, because each pair appears once).

```python
back = long.pivot(index="store_id", columns="category", values="revenue")
assert back.equals(grid)
```

`stack` and `unstack` do the same kind of reshape using the index. `unstack` moves the inner level of a MultiIndex into the columns, so the grouped Series from 8.3 becomes the grid. `stack` moves columns back into the index.

```python
from_groupby = sc.unstack()
print(from_groupby.equals(grid))
print(grid.stack().head(4))
```

Use whichever fits: `pivot_table` and `melt` when you start from columns, `unstack` and `stack` when the keys are already in the index.

## 8.9 Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Expecting a plain table after grouping by two keys | You get a MultiIndex that is awkward to merge or save | `.reset_index()` or `groupby(..., as_index=False)` |
| Using `agg` when you need a value per row | The result has one row per group and will not line up | Use `transform` |
| Default inner `merge` with incomplete lookup tables | Unmatched rows vanish without warning | `how="left"` plus `indicator=True`, then check |
| Merging on a key that is not unique in the right table | Rows are duplicated and totals inflate | `validate="many_to_one"` |
| `concat` without `ignore_index=True` | Repeated index labels, confusing `.loc` results | `pd.concat([a, b], ignore_index=True)` |
| `pivot` when a pair appears more than once | `ValueError: Index contains duplicate entries` | Use `pivot_table` with an `aggfunc` |

## 8.10 Cheat sheet

| Task | Code |
|---|---|
| Sum per group | `df.groupby("store_id")["line_total"].sum()` |
| Named aggregations | `df.groupby("k").agg(total=("x", "sum"), n=("x", "size"))` |
| Several keys, flat result | `df.groupby(["a", "b"], as_index=False)["x"].sum()` |
| Group value on every row | `df.groupby("k")["x"].transform("sum")` |
| Keep whole groups | `df.groupby("k").filter(lambda g: len(g) >= 70)` |
| Join on a key | `a.merge(b, on="key", how="left")` |
| Audit / check a join | `indicator=True`, `validate="many_to_one"` |
| Stack tables vertically | `pd.concat([a, b], ignore_index=True)` |
| Pivot table | `df.pivot_table(index="a", columns="b", values="x", aggfunc="sum")` |
| Count combinations | `pd.crosstab(df["a"], df["b"], normalize="index")` |
| Wide to long | `wide.melt(id_vars="a", var_name="b", value_name="x")` |
| Long to wide | `long.pivot(index="a", columns="b", values="x")` |
| Index level to columns / back | `s.unstack()`, `df.stack()` |
| MultiIndex to columns | `df.reset_index()` |

## 8.11 Exercises

**1.** For each payment method, compute the number of order lines, the total revenue and the average line total, with clear column names. Which method brings in the most revenue?

```python
# Solution
pay = orders.groupby("payment").agg(
    lines=("line_total", "size"),
    revenue=("line_total", "sum"),
    avg_line=("line_total", "mean"),
).round(1)
print(pay.sort_values("revenue", ascending=False))
```

**2.** Add a column `item_share` to `orders` giving each line's share of the total revenue of **its item**. Check that the shares of each item add up to 1.

```python
# Solution
orders["item_share"] = orders["line_total"] / orders.groupby("item")["line_total"].transform("sum")
check = orders.groupby("item")["item_share"].sum()
assert np.allclose(check, 1)
print(check.round(6).head(3))
```

**3.** Join `orders` with `customers` to find the total revenue per **home store** of the customer (ignore guest orders). Use `validate=` to make sure no rows are duplicated.

```python
# Solution
m = orders.dropna(subset=["customer_id"]).merge(
    customers[["customer_id", "home_store"]], on="customer_id", validate="many_to_one")
print(m.groupby("home_store")["line_total"].sum())
```

**4.** Build a pivot table of the **number of distinct orders** per store (rows) and payment method (columns). Then melt it into a long table with columns `store_id`, `payment` and `orders`.

```python
# Solution
pv = orders.pivot_table(index="store_id", columns="payment", values="order_id", aggfunc="nunique")
long_pv = pv.reset_index().melt(id_vars="store_id", var_name="payment", value_name="orders")
print(pv)
print(long_pv.head(4))
```
