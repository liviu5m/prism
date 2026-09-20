import sqlite3
from faker import Faker
import random
from datetime import timedelta
conn = sqlite3.connect("core.db")
fake = Faker()
cursor = conn.cursor()
cursor.execute("ATTACH DATABASE 'cars.db' AS cars;")
cursor.execute("ATTACH DATABASE 'orders.db' AS orders;")
cursor.execute("PRAGMA foreign_keys = ON;")
cursor.executescript(
    """
    DROP TABLE IF EXISTS orders.user_orders;
    DROP TABLE IF EXISTS orders.item;
    DROP TABLE IF EXISTS orders.shop;
    DROP TABLE IF EXISTS orders.category;
    DROP TABLE IF EXISTS orders.user;
    CREATE TABLE orders.user(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        created_at TEXT
    );

    CREATE TABLE orders.category(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    );

    CREATE TABLE orders.shop(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        address TEXT NOT NULL,
        created_at TEXT
    );

    CREATE TABLE orders.item(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category_id INTEGER,
        shop_id INTEGER,
        price INTEGER NOT NULL,
        stock INTEGER NOT NULL,
        FOREIGN KEY(category_id) REFERENCES category(id),
        FOREIGN KEY(shop_id) REFERENCES shop(id)
    );

    CREATE TABLE orders.user_orders(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        item_id INTEGER,
        quantity INTEGER NOT NULL,
        created_at TEXT,
        FOREIGN KEY(user_id) REFERENCES user(id),
        FOREIGN KEY(item_id) REFERENCES item(id)
    );

    DROP TABLE IF EXISTS cars.maintenance_log;
    DROP TABLE IF EXISTS cars.rental;
    DROP TABLE IF EXISTS cars.car;
    DROP TABLE IF EXISTS cars.dealership;
    DROP TABLE IF EXISTS cars.user;
    CREATE TABLE cars.user(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phone TEXT NOT NULL UNIQUE,
        created_at TEXT
    );

    CREATE TABLE cars.dealership(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        address TEXT NOT NULL,
        city TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE cars.car(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dealership_id INTEGER NOT NULL,
        make TEXT NOT NULL,
        model TEXT NOT NULL,
        year INTEGER NOT NULL,
        vin TEXT NOT NULL UNIQUE,
        daily_rate REAL NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('available', 'rented', 'maintenance')),
        FOREIGN KEY (dealership_id) REFERENCES dealership(id)
    );

    CREATE TABLE cars.rental(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        car_id INTEGER NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        total_price REAL NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user(id),
        FOREIGN KEY (car_id) REFERENCES car(id)
    );

    CREATE TABLE cars.maintenance_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        car_id INTEGER NOT NULL,
        service_details TEXT NOT NULL,
        cost REAL NOT NULL,
        serviced_at TEXT NOT NULL,
        FOREIGN KEY (car_id) REFERENCES car(id)
    );

    """
)

users_data = []

for i in range(30):
    users_data.append(
        (fake.name(), fake.email(), fake.date_time().isoformat())
    )

cursor.executemany("INSERT INTO orders.user(name, email, created_at) VALUES (?, ?, ?)", users_data)
categories_data = []
CATEGORIES = [
    "Electronics",
    "Books & Stationery",
    "Clothing & Apparel",
    "Home & Kitchen",
    "Sports & Outdoors",
    "Beauty & Personal Care",
    "Toys & Games",
    "Automotive",
    "Groceries",
    "Health & Wellness",
]

categories_data = [(cat,) for cat in CATEGORIES]
cursor.executemany("INSERT INTO orders.category(name) VALUES (?)", categories_data)
shops_data = []
SHOPS = [
    "Amazon",
    "Walmart",
    "Target",
    "Best Buy",
    "Kroger",
    "Home Depot",
    "Lowe's",
    "Macy's",
    "Meijer",
    "Newegg",
    "Walgreens",
    "Whole Foods",
]

shops_data = [(shop,fake.address()) for shop in SHOPS]
cursor.executemany("INSERT INTO orders.shop(name, address) VALUES (?, ?)", shops_data)

items_data = []
for i in range(100):
    items_data.append(
        (
            fake.catch_phrase(),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(10, 100),
            random.randint(1, 10),
        )
    )
cursor.executemany(
    "INSERT INTO orders.item(name, category_id, shop_id, price, stock) VALUES (?,?,?, ?, ?)",
    items_data,
)

orders_data = []
for i in range(100):
    orders_data.append(
        (
            random.randint(1, 30),
            random.randint(1, 100),
            random.randint(1, 10),
            fake.date_time().isoformat(),
        )
    )
conn.commit()
cursor.executemany(
    "INSERT INTO orders.user_orders(user_id, item_id, quantity, created_at) VALUES (?, ?, ?, ?)",
    orders_data,
)



users_data = []
for _ in range(20):
    users_data.append(
        (
            fake.name(),
            fake.unique.email(),
            fake.unique.phone_number(),
            fake.date_time_this_year().isoformat(),
        )
    )

cursor.executemany(
    "INSERT INTO cars.user (name, email, phone, created_at) VALUES (?, ?, ?, ?)",
    users_data,
)

# 2. Insert 5 Dealerships
dealerships_data = []
for _ in range(5):
    dealerships_data.append(
        (
            f"{fake.company()} Motors",
            fake.street_address(),
            fake.city(),
            fake.date_time_this_year().isoformat(),
        )
    )

cursor.executemany(
    "INSERT INTO cars.dealership (name, address, city, created_at) VALUES (?, ?, ?, ?)",
    dealerships_data,
)

MAKES_MODELS = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot"],
    "Ford": ["Mustang", "F-150", "Explorer", "Focus"],
    "BMW": ["3 Series", "5 Series", "X5", "M4"],
    "Tesla": ["Model 3", "Model Y", "Model S"],
}

cars_data = []
for _ in range(30):
    make = random.choice(list(MAKES_MODELS.keys()))
    model = random.choice(MAKES_MODELS[make])
    cars_data.append(
        (
            random.randint(1, 5), 
            make,
            model,
            random.randint(2020, 2026),  # year
            fake.unique.bothify(text="??#?#??##?######").upper(),  # VIN
            round(random.uniform(40.0, 250.0), 2),  # daily_rate
            random.choice(["available", "rented", "maintenance"]),  # status
        )
    )

cursor.executemany(
    """
    INSERT INTO cars.car (dealership_id, make, model, year, vin, daily_rate, status) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    cars_data,
)

# 4. Insert 40 Rentals
rentals_data = []
for _ in range(40):
    start_date = fake.date_time_this_year()
    end_date = fake.date_time_between(start_date=start_date, end_date=start_date + timedelta(days=random.randint(1, 30)))
    days = max((end_date - start_date).days, 1)
    daily_rate = round(random.uniform(45.0, 150.0), 2)

    rentals_data.append(
        (
            random.randint(1, 20),  # user_id
            random.randint(1, 30),  # car_id
            start_date.isoformat(),
            end_date.isoformat(),
            round(days * daily_rate, 2),  # total_price
        )
    )

cursor.executemany(
    """
    INSERT INTO cars.rental (user_id, car_id, start_date, end_date, total_price) 
    VALUES (?, ?, ?, ?, ?)
    """,
    rentals_data,
)

# 5. Insert 15 Maintenance Logs
SERVICES = [
    "Oil Change & Filter Replacement",
    "Brake Pad Replacement",
    "Tire Rotation & Balancing",
    "Transmission Fluid Flush",
    "Battery Replacement",
]

maintenance_data = []
for _ in range(15):
    maintenance_data.append(
        (
            random.randint(1, 30),  # car_id
            random.choice(SERVICES),
            round(random.uniform(80.0, 600.0), 2),  # cost
            fake.date_time_this_year().isoformat(),
        )
    )

cursor.executemany(
    """
    INSERT INTO cars.maintenance_log (car_id, service_details, cost, serviced_at) 
    VALUES (?, ?, ?, ?)
    """,
    maintenance_data,
)
conn.commit()
conn.close()
