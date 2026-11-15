"""Category taxonomy. Loosely modeled on Plaid's primary personal-finance categories."""

CATEGORIES: tuple[str, ...] = (
    "GROCERIES",
    "DINING",
    "COFFEE",
    "TRANSPORT",
    "GAS",
    "TRAVEL",
    "SHOPPING",
    "SUBSCRIPTIONS",
    "UTILITIES",
    "ENTERTAINMENT",
    "HEALTH",
    "FITNESS",
    "EDUCATION",
    "HOUSING",
    "LOANS",
    "TRANSFER",
    "INCOME",
    "FEES",
)

UNKNOWN = "UNKNOWN"

DESCRIPTIONS: dict[str, str] = {
    "GROCERIES": "Supermarkets, grocery delivery",
    "DINING": "Restaurants, fast food, food delivery",
    "COFFEE": "Coffee shops, tea, boba",
    "TRANSPORT": "Rideshare, transit, parking, tolls, scooters",
    "GAS": "Gas stations, EV charging",
    "TRAVEL": "Airlines, hotels, rentals, trains, car rental",
    "SHOPPING": "Retail, online marketplaces, clothing, home goods",
    "SUBSCRIPTIONS": "Streaming, software, apps, digital memberships",
    "UTILITIES": "Phone, internet, electric, gas, water, trash",
    "ENTERTAINMENT": "Movies, concerts, events, games",
    "HEALTH": "Pharmacy, doctors, dental, therapy",
    "FITNESS": "Gyms, classes, fitness apps/hardware",
    "EDUCATION": "Tuition, textbooks, courses",
    "HOUSING": "Rent, mortgage, renters/home insurance",
    "LOANS": "Student/personal/auto loans, BNPL installments",
    "TRANSFER": "P2P, card payments, brokerage, ATM withdrawals",
    "INCOME": "Payroll, tax refunds, interest",
    "FEES": "Bank fees: overdraft, ATM, maintenance, foreign txn",
}

assert set(DESCRIPTIONS) == set(CATEGORIES)
