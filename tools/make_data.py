"""Generate the practice datasets in data/ (seeded, so every run gives the same files).

A small chain of coffee shops, "Brew & Byte", over 2024:
- stores.csv     4 stores with city and opening year
- customers.csv  loyalty-card customers (with some messy values for Chapter 7)
- orders.csv     one row per order line, 2024-01-01 to 2024-12-31
- weather.csv    daily temperature and rain per city
- orders_messy.csv  a deliberately dirty copy of part of orders.csv for cleaning practice
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
rng = np.random.default_rng(2024)

stores = pd.DataFrame({
    "store_id": ["S1", "S2", "S3", "S4"],
    "city": ["Pune", "Pune", "Mumbai", "Bengaluru"],
    "opened": [2018, 2021, 2019, 2022],
    "seats": [40, 24, 55, 32],
})

menu = pd.DataFrame({
    "item": ["Espresso", "Cappuccino", "Latte", "Cold Brew", "Masala Chai", "Green Tea",
             "Croissant", "Muffin", "Sandwich", "Cookie"],
    "category": ["Coffee", "Coffee", "Coffee", "Coffee", "Tea", "Tea",
                 "Bakery", "Bakery", "Food", "Bakery"],
    "price": [120, 160, 180, 200, 90, 110, 140, 110, 220, 70],
    "popularity": [8, 14, 16, 7, 12, 5, 8, 7, 9, 10],
})

n_customers = 600
first = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vihaan", "Isha", "Arjun", "Sara",
         "Neel", "Tara", "Dev", "Riya", "Kiran", "Zoya", "Om", "Pia", "Yash", "Nina"]
customers = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in range(1, n_customers + 1)],
    "name": rng.choice(first, n_customers),
    "age": rng.integers(17, 66, n_customers),
    "home_store": rng.choice(stores.store_id, n_customers, p=[0.3, 0.2, 0.3, 0.2]),
    "joined": pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 1000, n_customers), unit="D"),
    "tier": rng.choice(["Bronze", "Silver", "Gold"], n_customers, p=[0.6, 0.3, 0.1]),
})
# a little realistic mess: missing ages and inconsistent tier spelling
customers.loc[rng.choice(n_customers, 25, replace=False), "age"] = np.nan
idx = rng.choice(n_customers, 20, replace=False)
customers.loc[idx, "tier"] = customers.loc[idx, "tier"].str.lower()
customers["age"] = customers["age"].astype("Int64")

days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
cities = stores.city.unique()
doy = np.arange(len(days))
base = {"Pune": 26, "Mumbai": 28, "Bengaluru": 24}
weather = []
for c in cities:
    temp = base[c] + 4 * np.sin(2 * np.pi * (doy - 60) / 365) + rng.normal(0, 1.5, len(days))
    monsoon = (days.month >= 6) & (days.month <= 9)
    rain = np.where(rng.random(len(days)) < np.where(monsoon, 0.7, 0.08),
                    rng.gamma(2.0, np.where(monsoon, 8.0, 2.0)), 0.0)
    weather.append(pd.DataFrame({"date": days, "city": c, "temp_c": temp.round(1), "rain_mm": rain.round(1)}))
weather = pd.concat(weather, ignore_index=True)

rows = []
order_id = 100000
weekday_boost = np.array([1.0, 0.95, 0.95, 1.0, 1.1, 1.35, 1.3])  # Mon..Sun
for _, s in stores.iterrows():
    w = weather[weather.city == s.city].set_index("date")
    for d in days:
        expected = s.seats * 0.6 * weekday_boost[d.day_of_week] * (1 + 0.15 * (d.month == 12))
        if w.loc[d, "rain_mm"] > 10:
            expected *= 0.8
        n_orders = rng.poisson(expected)
        hours = rng.choice(np.arange(7, 22), n_orders,
                           p=np.array([3, 8, 10, 7, 5, 6, 7, 5, 4, 5, 6, 5, 4, 3, 2]) / 80)
        for h in hours:
            order_id += 1
            cust = rng.choice(customers.customer_id) if rng.random() < 0.55 else ""
            pop = menu.popularity.to_numpy(dtype=float).copy()
            if w.loc[d, "temp_c"] > 29:
                pop[3] *= 3  # cold brew on hot days
            k = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            items = rng.choice(len(menu), k, replace=False, p=pop / pop.sum())
            ts = d + pd.Timedelta(hours=int(h), minutes=int(rng.integers(0, 60)))
            for it in items:
                rows.append((order_id, ts, s.store_id, cust, menu.item[it], menu.category[it],
                             int(rng.choice([1, 1, 1, 2])), menu.price[it]))

orders = pd.DataFrame(rows, columns=["order_id", "timestamp", "store_id", "customer_id",
                                     "item", "category", "qty", "unit_price"])
orders["payment"] = rng.choice(["UPI", "Card", "Cash"], len(orders), p=[0.6, 0.25, 0.15])

# messy sample for the cleaning chapter
messy = orders.sample(1500, random_state=7).reset_index(drop=True).copy()
messy["timestamp"] = messy["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
messy = messy.astype({"qty": "object", "unit_price": "object"})
messy.loc[rng.choice(1500, 40, replace=False), "qty"] = np.nan
messy.loc[rng.choice(1500, 15, replace=False), "unit_price"] = "Rs. 180"
latte_rows = messy.index[messy["item"] == "Latte"]
messy.loc[rng.choice(latte_rows, min(30, len(latte_rows)), replace=False), "item"] = "  latte "
messy.loc[rng.choice(1500, 12, replace=False), "store_id"] = "s3"
messy.loc[rng.choice(1500, 5, replace=False), "qty"] = 40  # suspicious bulk orders
messy = pd.concat([messy, messy.sample(25, random_state=3)], ignore_index=True)  # duplicates

DATA.mkdir(exist_ok=True)
stores.to_csv(DATA / "stores.csv", index=False)
menu.drop(columns="popularity").to_csv(DATA / "menu.csv", index=False)
customers.to_csv(DATA / "customers.csv", index=False)
weather.to_csv(DATA / "weather.csv", index=False)
orders.to_csv(DATA / "orders.csv", index=False)
messy.to_csv(DATA / "orders_messy.csv", index=False)
for f in sorted(DATA.glob("*.csv")):
    print(f"{f.name:20s} {sum(1 for _ in open(f, encoding='utf-8')) - 1:>7} rows  {f.stat().st_size/1e6:.2f} MB")
