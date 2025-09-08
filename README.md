# TLCO (Total Lifecycle Cost of Ownership) ToolkitA python app that calculates the total lifecycle cost of ownership for a car. Let's you actually determine what a car will cost you over the time you own it, including gas, insurance, maintenance, depreciation, etc. instead of just comparing sticker price.



Compute TLCO across multiple car options defined in `./cars.json`.
Containerized for reproducible runs. Results are written to `./data/`.

## Quick start

```bash
# 1) Create this repository structure and paste the files from the README.
# 2) Edit cars.json with your real numbers (year/model/trim, quotes, insurance, etc.)
# 3) Build & run (results will land in ./data/)
docker compose up --build
```

## Depeciation rates for older cars

How to use it—2023 Sienna example (buy in 2025, 3-yr horizon)

Current age ≈ 2 years, so use next 3 ages: Y3, Y4, Y5 → 10%, 7%, 6%.

Compute forward retention factor = Π(1 − rate):
0.90 × 0.93 × 0.94 = 0.78678

Forward total drop = 1 − 0.78678 = 21.32%

Apply mileage multiplier (+2.4%):
21.32% × 1.024 = 21.83% total over the next 3 years

What to put in your JSON (two options):
"total_depreciation_pct_over_horizon": 0.218

