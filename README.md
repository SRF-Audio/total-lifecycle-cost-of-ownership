# TLCO (Total Lifecycle Cost of Ownership) ToolkitA python app that calculates the total lifecycle cost of ownership for a car. Let's you actually determine what a car will cost you over the time you own it, including gas, insurance, maintenance, depreciation, etc. instead of just comparing sticker price.



Compute TLCO across multiple car options defined in `./cars.json`.
Containerized for reproducible runs. Results are written to `./data/`.

## Quick start

```bash
# 1) Create this repository structure and paste the files from the README.
# 2) Edit cars.json with your real numbers (year/model/trim, quotes, insurance, etc.)
# 3) Build & run (results will land in ./data/)
docker compose up --build
