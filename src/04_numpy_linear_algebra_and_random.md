# Chapter 4: Linear algebra and random simulation with NumPy

This chapter covers two toolkits that power a lot of data science behind the scenes. **Linear algebra** is the maths of tables of numbers: multiplying them, solving equations with them, and fitting a straight line through data. **Random simulation** lets you answer "what usually happens?" questions by playing a situation out thousands of times on the computer.

You do not need any maths beyond school level here. By the end you will be able to multiply matrices and predict the result's shape, solve a small system of equations, fit a trend line to real sales data, draw samples from common distributions, and run a reproducible Monte Carlo simulation. It builds on Chapter 3 (aggregations, `axis` and broadcasting) and the random Generator from 1.4.

```python
import numpy as np

rng = np.random.default_rng(42)
```

## 4.1 Dot products and matrix multiplication

The **dot product** of two equal-length vectors multiplies them element by element and adds up the results. You have already computed one: a bill is quantities times prices, summed. NumPy writes it as `np.dot(a, b)` or, more readably, `a @ b`.

```python
qty = np.array([2, 1, 3])            # Latte, Croissant, Cookie
price = np.array([180, 140, 70])

print((qty * price).sum())           # the long way (see 3.1 and 3.2)
print(qty @ price)                   # the dot product
```

A **matrix product** is many dot products at once. If each row of a matrix is one customer's basket, then `baskets @ price` gives one bill per customer.

```python
baskets = np.array([[2, 1, 3],       # customer A
                    [0, 2, 1],       # customer B
                    [1, 0, 0]])      # customer C
print(baskets @ price)
```

The shape rule is the one thing to memorise: `(m, n) @ (n, p)` gives `(m, p)`. The **inner** sizes must match (they are what gets multiplied and summed) and the **outer** sizes form the result.

```
baskets   (3, 3)      3 customers x 3 items
price        (3,)     3 items
result    (3,)        one bill per customer
```

Now give each item a price at two stores. The price table has shape `(3 items, 2 stores)`, and the result is a bill for every customer at every store:

```python
store_price = np.array([[180, 190],     # Latte at S1, S3
                        [140, 150],     # Croissant
                        [ 70,  60]])    # Cookie
bills = baskets @ store_price           # (3, 3) @ (3, 2) -> (3, 2)
print(bills)
```

When the inner sizes do not line up, the **transpose** `.T` (see 1.6) often fixes it by swapping rows and columns. Remember that `@` is not the same as `*`: `*` multiplies element by element with broadcasting (see 3.5), while `@` does sums of products.

```python
print(store_price.shape, store_price.T.shape)
try:
    baskets @ store_price.T             # (3, 3) @ (2, 3): inner sizes 3 and 2 clash
except ValueError as e:
    print("ValueError:", str(e)[:70], "...")
```

## 4.2 Solving a system of equations

Matrices let you work backwards too. Suppose you lost the price list but kept three receipts:

- 2 espressos + 1 cappuccino + 1 croissant = 540
- 1 espresso + 2 cappuccinos = 440
- 1 espresso + 1 cappuccino + 2 croissants = 560

Write the quantities as a matrix `A` and the totals as a vector `b`. You are looking for the prices `x` such that `A @ x == b`. `np.linalg.solve` finds them.

```python
A = np.array([[2, 1, 1],
              [1, 2, 0],
              [1, 1, 2]])
b = np.array([540, 440, 560])

x = np.linalg.solve(A, b)
print(x)                       # espresso, cappuccino, croissant
assert np.allclose(A @ x, b)   # the prices reproduce every receipt
```

The prices are 120, 160 and 140, which match `data/menu.csv`. You may also see this written with the **inverse** matrix, `np.linalg.inv(A) @ b`. It gives the same answer but is slower and less accurate on big problems, so prefer `solve`.

```python
print(np.linalg.inv(A) @ b)
```

Not every system has a single answer. If one receipt is just a multiple of another, the information is redundant. The **determinant** (`np.linalg.det`) is 0 in that case, and `solve` raises an error.

```python
print(np.linalg.det(A).round(2))

B = np.array([[1, 2], [2, 4]])          # second row is twice the first
print(np.linalg.det(B))
try:
    np.linalg.solve(B, np.array([300, 600]))
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)
```

## 4.3 Norms, distances and eigenvalues

The **norm** of a vector is its length. The default `np.linalg.norm` is the everyday straight-line length (Pythagoras). The norm of a **difference** is the distance between two points, which is how many machine learning methods decide that two rows are "similar".

```python
print(np.linalg.norm(np.array([3, 4])))       # 5, the classic 3-4-5 triangle

# daily averages for three stores: [cups of coffee, cups of tea, bakery items]
s1 = np.array([60, 25, 30])
s2 = np.array([55, 28, 26])
s4 = np.array([20, 40, 10])
print(np.linalg.norm(s1 - s2).round(1), np.linalg.norm(s1 - s4).round(1))
```

S1 is much closer to S2 than to S4: their sales mixes look alike. With `axis` you can measure many rows at once, here the distance of every store from S1:

```python
profiles = np.array([s1, s2, s4])
print(np.linalg.norm(profiles - s1, axis=1).round(1))   # broadcasting, then one length per row
```

**Eigenvalues** and **eigenvectors** describe the directions in which a matrix simply stretches things. You will meet them properly in the machine learning stage (they are the engine of PCA); for now it is enough to know where they live. `np.linalg.eig` returns the stretch factors and the directions.

```python
M = np.array([[2.0, 0.0],
              [0.0, 0.5]])
values, vectors = np.linalg.eig(M)
print(values)       # stretch by 2 along x, shrink by half along y
print(vectors)      # the directions, one per column
```

## 4.4 Fitting a line with least squares

Here is linear algebra doing real work. Does hot weather sell more Cold Brew in Mumbai? We will fit a straight line `sales = a * temperature + b` through a year of daily data from `data/orders.csv` and `data/weather.csv`.

First, total the Cold Brew quantity per day at the Mumbai store (S3). We turn each timestamp into a day number (0 to 365) and let `np.bincount` add up the quantities per day. Dates get a full chapter of their own later (Chapter 9).

```python
raw = np.loadtxt("data/orders.csv", delimiter=",", skiprows=1, usecols=(1, 2, 4, 6), dtype=str)
day = (raw[:, 0].astype("datetime64[D]") - np.datetime64("2024-01-01")).astype(int)
qty_all = raw[:, 3].astype(int)

is_cold_brew_s3 = (raw[:, 1] == "S3") & (raw[:, 2] == "Cold Brew")
cold_brew = np.bincount(day[is_cold_brew_s3], weights=qty_all[is_cold_brew_s3], minlength=366)
print(cold_brew.shape, cold_brew[:7])
```

`weights=` makes `bincount` add up the quantities instead of just counting rows. Next, Mumbai's daily temperature (rows 366 to 731 of the weather file, see 2.3):

```python
mumbai_t = np.loadtxt("data/weather.csv", delimiter=",", skiprows=1, usecols=2)[366:732]
print(mumbai_t.shape, mumbai_t.min(), mumbai_t.max())
```

A straight line through 366 points cannot pass through all of them, so there is no exact solution as in 4.2. **Least squares** finds the line that makes the squared vertical misses as small as possible. In matrix form we want `X @ [a, b] ≈ y`, where `X` has one column of temperatures and one column of 1s (the 1s let the line have an intercept `b`).

```python
X = np.column_stack([mumbai_t, np.ones_like(mumbai_t)])    # (366, 2)
y = cold_brew                                               # (366,)

coef, residuals, rank, sv = np.linalg.lstsq(X, y)
a, b = coef
print(f"slope a = {a:.2f} cups per °C, intercept b = {b:.2f}")
```

The slope is the useful number: on average each extra degree goes with that many more cups of Cold Brew a day. We can use the line to predict, and measure how good it is with **R²** (the share of the day-to-day variation the line explains: 0 is useless, 1 is perfect).

```python
pred = X @ coef
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("R² =", round(r2, 2))

for t in [25, 30, 34]:
    print(f"{t} °C -> about {a * t + b:.1f} Cold Brews")
```

The fit captures a clear trend, but a lot of the daily variation remains unexplained, which is normal for real sales: weekends, holidays and plain chance matter too. Chapter 12 measures this temperature and Cold Brew relationship properly with correlation tests (see 12.6), and the project in Chapter 13 shows why a straight line is not the whole story (see 13.9).

## 4.5 The random Generator and its distributions

You met `rng = np.random.default_rng(42)` in 1.4. The Generator can draw from many **distributions**: rules that say which values are likely. Picking the right one is half of building a good simulation.

| Method | Gives | Coffee-shop example |
|---|---|---|
| `rng.integers(lo, hi, size)` | whole numbers, all equally likely | a random table number |
| `rng.random(size)` | decimals in [0, 1) | a probability check |
| `rng.uniform(lo, hi, size)` | decimals, all equally likely in a range | a delivery time between 20 and 40 min |
| `rng.normal(mean, sd, size)` | bell curve around a mean | the weight of a muffin |
| `rng.poisson(lam, size)` | counts of events in a time window | customers arriving per minute |
| `rng.binomial(n, p, size)` | successes out of n tries | how many of 50 customers pay by UPI |
| `rng.exponential(scale, size)` | waiting times between events | minutes until the next customer |

```python
print(rng.uniform(20, 40, size=3).round(1))     # delivery minutes
print(rng.normal(110, 5, size=3).round(1))      # muffin grams
print(rng.poisson(2.5, size=10))                # arrivals in 10 separate minutes
print(rng.binomial(50, 0.6, size=5))            # UPI payers out of 50, five times
```

A quick sanity check is to draw many values and compare the sample with what the distribution promises. The mean of a Poisson distribution is `lam`, and its variance is also `lam`:

```python
arrivals = rng.poisson(2.5, size=100_000)
print(arrivals.mean().round(3), arrivals.var().round(3))
print(np.bincount(arrivals)[:8])        # how often 0, 1, 2, ... customers arrived
```

The `size` argument accepts a shape, so one call can fill a whole table, for example 4 stores x 7 days of arrivals (see 1.4).

```python
print(rng.poisson(lam=[[60], [35], [80], [45]], size=(4, 7)))   # each row has its own rate, by broadcasting
```

## 4.6 Shuffling and sampling

Data work constantly needs random **orderings** and random **subsets**: shuffling rows before splitting into training and test sets, picking a sample of customers for a survey, or drawing items with given popularities.

`rng.permutation` returns a shuffled copy; `rng.shuffle` shuffles in place and returns nothing.

```python
ids = np.arange(10)
print(rng.permutation(ids))   # new shuffled array
print(ids)                    # original untouched

rng.shuffle(ids)              # shuffles ids itself
print(ids)
```

A common pattern: shuffle **positions**, then use them to index several arrays the same way (fancy indexing, see 2.7). This keeps rows lined up, for example to split data 80/20.

```python
n = 10
order = rng.permutation(n)
train_idx, test_idx = order[:8], order[8:]
print(train_idx, test_idx)
```

`rng.choice` samples from an array. By default it samples **with replacement** (the same element can come up twice). Pass `replace=False` to sample **without replacement**, like drawing raffle tickets from a hat, and `p=` to give each element its own probability.

```python
customers = np.array(["C0001", "C0002", "C0003", "C0004", "C0005", "C0006"])
print(rng.choice(customers, size=3, replace=False))     # 3 different survey winners

items = np.array(["Latte", "Cappuccino", "Masala Chai", "Cookie"])
popularity = [0.4, 0.3, 0.2, 0.1]                         # must add up to 1
picks = rng.choice(items, size=1000, p=popularity)
print(np.unique(picks, return_counts=True))
```

With 1,000 draws the counts land close to 400, 300, 200 and 100. `np.unique(..., return_counts=True)` returns the distinct values and how often each appears; note it lists them alphabetically.

## 4.7 Monte Carlo simulation

A **Monte Carlo** simulation answers a question by random trial: play the situation out many times and look at the results. It is the go-to tool when the maths is hard but the rules are easy to describe.

### Warm-up: estimating π

Throw random points into a 1 x 1 square. The fraction that land inside the quarter circle of radius 1 is close to π/4, so four times that fraction estimates π.

```python
n = 1_000_000
pts = rng.random((n, 2))                       # x and y in [0, 1)
inside = (pts ** 2).sum(axis=1) <= 1           # x² + y² <= 1
pi_hat = 4 * inside.mean()
print(pi_hat, f"error: {abs(pi_hat - np.pi):.6f}")
```

More points give a better estimate, but slowly: to get one extra correct digit you need about 100 times as many points.

```python
for n in [100, 10_000, 1_000_000]:
    p = rng.random((n, 2))
    print(f"{n:>9,} points -> {4 * ((p ** 2).sum(axis=1) <= 1).mean():.4f}")
```

### A morning rush at the counter

Now a real question for Brew & Byte. During the 8:00 to 10:00 rush, about 3 customers arrive per minute on average (Poisson, see 4.5). One barista serves 3 customers per minute at best. How long does the queue get, and does a second, slower trainee barista (2 more per minute) help?

The rules per minute: new customers join the queue, then the staff serve as many as they can. We simulate 5,000 mornings **at once**, one per row, and loop only over the 120 minutes.

```python
def simulate_rush(serve_per_min, n_runs=5000, minutes=120, rate=3.0, seed=7):
    g = np.random.default_rng(seed)
    queue = np.zeros(n_runs)
    longest = np.zeros(n_runs)
    for _ in range(minutes):
        queue += g.poisson(rate, size=n_runs)            # arrivals this minute
        queue = np.maximum(queue - serve_per_min, 0)     # served, never below 0
        longest = np.maximum(longest, queue)
    return longest
```

```python
one = simulate_rush(serve_per_min=3)
two = simulate_rush(serve_per_min=5)

for name, res in [("1 barista", one), ("2 baristas", two)]:
    print(f"{name}: average longest queue {res.mean():.1f}, "
          f"90th percentile {np.percentile(res, 90):.0f}, "
          f"P(queue ever > 10) = {(res > 10).mean():.2f}")
```

With one barista the service rate only just equals the arrival rate, so random bursts pile up and the queue regularly gets long. Adding the trainee keeps it short almost every morning. Notice how the answer is a **distribution** of outcomes (average, 90th percentile, a probability), not one number: that is the main payoff of simulation.

## 4.8 Reproducibility and seeds

A random simulation that gives different numbers every run is hard to debug and impossible to check. A **seed** fixes the starting point of the generator, so the same code gives the same numbers on every machine. That is why the setup cell uses `default_rng(42)` and `simulate_rush` takes a `seed`.

```python
g1 = np.random.default_rng(2024)
g2 = np.random.default_rng(2024)
print(g1.integers(0, 100, 5))
print(g2.integers(0, 100, 5))        # same seed, same numbers
```

A generator keeps moving forward as you draw from it, so the same call twice gives different numbers. Results also depend on **what you drew before**: inserting one extra cell above a simulation changes everything after it. For important results, create a dedicated generator with its own seed right where you need it.

```python
g = np.random.default_rng(2024)
print(g.integers(0, 100, 5))
print(g.integers(0, 100, 5))         # continues the stream: different values

print(np.array_equal(simulate_rush(3), simulate_rush(3)))   # own seed inside: always identical
```

When you run several independent simulations (for example one per store), `spawn` gives each its own generator whose numbers do not overlap with the others, all from one parent seed:

```python
parent = np.random.default_rng(99)
store_rngs = parent.spawn(4)
print([int(r.poisson(60)) for r in store_rngs])      # one morning's arrivals per store
```

Choose any seed you like; there is nothing special about 42. What matters is that you choose one and write it down.

## 4.9 Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Using `*` when you meant a matrix product | Element-wise result (or broadcast error), not sums of products | Use `@` (or `np.dot`) |
| Inner sizes do not match in `@` | `ValueError: matmul: Input operand 1 has a mismatch...` | Check shapes; `(m, n) @ (n, p)`. Try `.T` |
| `np.linalg.inv(A) @ b` for every system | Slower and less accurate | `np.linalg.solve(A, b)` |
| Forgetting the column of 1s in a line fit | The line is forced through 0 and the slope is wrong | `np.column_stack([x, np.ones_like(x)])` |
| `rng.choice` with `p` that does not sum to 1 | `ValueError: probabilities do not sum to 1` | Normalise: `p = w / w.sum()` |
| `x = rng.shuffle(x)` | `x` becomes `None`, because `shuffle` works in place | `rng.shuffle(x)` alone, or `x = rng.permutation(x)` |
| Using `np.random.seed` / no seed at all | Results change run to run | `rng = np.random.default_rng(seed)` |

## 4.10 Cheat sheet

| Task | Code |
|---|---|
| Dot / matrix product | `a @ b` |
| Transpose | `m.T` |
| Solve `A x = b` | `np.linalg.solve(A, b)` |
| Inverse / determinant | `np.linalg.inv(A)`, `np.linalg.det(A)` |
| Length of a vector / row lengths | `np.linalg.norm(v)`, `np.linalg.norm(m, axis=1)` |
| Eigenvalues and vectors | `vals, vecs = np.linalg.eig(M)` |
| Least-squares line fit | `coef, *_ = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y)` |
| Seeded generator | `rng = np.random.default_rng(42)` |
| Normal / Poisson / binomial | `rng.normal(m, sd, n)`, `rng.poisson(lam, n)`, `rng.binomial(k, p, n)` |
| Shuffled copy / in place | `rng.permutation(a)`, `rng.shuffle(a)` |
| Sample without replacement | `rng.choice(a, size=3, replace=False)` |
| Weighted sample | `rng.choice(a, size=100, p=probs)` |
| Count distinct values | `np.unique(a, return_counts=True)` |
| Independent child generators | `rng.spawn(4)` |

## 4.11 Exercises

**1.** A customer buys 2 Lattes, 1 Muffin and 2 Cookies. Using the prices 180, 110 and 70, compute the bill with `@`. Then compute the bills for the three baskets `[[2, 1, 2], [1, 1, 0], [0, 3, 1]]` in one expression.

```python
# Solution
p = np.array([180, 110, 70])
print(np.array([2, 1, 2]) @ p)
print(np.array([[2, 1, 2], [1, 1, 0], [0, 3, 1]]) @ p)
```

**2.** Two receipts: "3 Masala Chai + 2 Green Tea = 490" and "1 Masala Chai + 4 Green Tea = 530". Find both prices with `np.linalg.solve` and check them with `assert`.

```python
# Solution
A2 = np.array([[3, 2], [1, 4]])
b2 = np.array([490, 530])
prices2 = np.linalg.solve(A2, b2)
print(prices2)
assert np.allclose(A2 @ prices2, b2)
```

**3.** Repeat the fit from 4.4 for **Masala Chai** at S3 against Mumbai's temperature. Is the slope positive or negative, and what does that suggest?

```python
# Solution
is_chai = (raw[:, 1] == "S3") & (raw[:, 2] == "Masala Chai")
chai = np.bincount(day[is_chai], weights=qty_all[is_chai], minlength=366)
coef_chai, *_ = np.linalg.lstsq(X, chai)
print(coef_chai.round(3))   # a negative slope: hot tea sells a little less on hot days
```

**4.** Simulate rolling two dice 100,000 times with `rng.integers` and estimate the probability that their total is 7. Compare it with the exact answer 6/36.

```python
# Solution
dice = rng.integers(1, 7, size=(100_000, 2))
p7 = (dice.sum(axis=1) == 7).mean()
print(round(p7, 4), round(6 / 36, 4))
```

**5.** Use `simulate_rush` to find the smallest whole number of customers served per minute (try 3 to 6) for which the chance of the queue ever exceeding 10 is below 5%.

```python
# Solution
for s in range(3, 7):
    risk = (simulate_rush(serve_per_min=s) > 10).mean()
    print(s, round(risk, 3))
    if risk < 0.05:
        print("smallest serving rate that works:", s)
        break
```
