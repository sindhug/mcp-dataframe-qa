# Zillow Metro Market Dataset

This repository includes `data/zillow_metro_market.csv`, a prepared public housing-market dataset built from Zillow Research for-sale listing time series.

## Why This Dataset

The project needs a real dataframe that is large enough to exercise dataframe QA patterns but small enough to clone, inspect, and query with local Pandas. The prepared CSV has:

- 91,872 rows
- 11 columns
- 928 Zillow Research geographies
- monthly observations from 2018-03-31 through 2026-05-31
- a 6.6 MB CSV footprint

This is intentionally not scraped individual listing data. It is aggregated public market data, which is a better default for an open repository: it is stable, compact, reproducible, and suitable for analytical questions about prices, active inventory, and new listings.

## Source

Source page: [Zillow Research Housing Data](https://www.zillow.com/research/data/)

Source files, accessed 2026-07-09:

- For-Sale Inventory, Metro & U.S., Smooth, All Homes, Monthly: `https://files.zillowstatic.com/research/public_csvs/invt_fs/Metro_invt_fs_uc_sfrcondo_sm_month.csv`
- New Listings, Metro & U.S., Smooth, All Homes, Monthly: `https://files.zillowstatic.com/research/public_csvs/new_listings/Metro_new_listings_uc_sfrcondo_sm_month.csv`
- Median List Price, Metro & U.S., Smooth, All Homes, Monthly: `https://files.zillowstatic.com/research/public_csvs/mlp/Metro_mlp_uc_sfrcondo_sm_month.csv`

Zillow Research describes for-sale inventory as the count of unique listings active at any time in a month, new listings as the count of listings that came on market in a month, and median list price as the median listed price across geographies.

## Transformation

Run:

```bash
python scripts/prepare_zillow_market_data.py
```

The script downloads the raw public CSVs into `data/raw/zillow/`, melts the wide date columns into a tidy long format, merges the three metrics, and writes:

```text
data/zillow_metro_market.csv
```

Raw downloads are ignored by Git. The prepared CSV is committed so the project works immediately after cloning.

## Columns

- `region_id`: Zillow Research region identifier.
- `size_rank`: Zillow Research region size rank.
- `region_name`: metro or national region name.
- `region_type`: Zillow geography type, such as `country` or `msa`.
- `state_name`: state abbreviation for the metro area, or `US` for the national row.
- `period`: month-end reporting date.
- `year`: calendar year extracted from `period`.
- `month`: calendar month extracted from `period`.
- `for_sale_inventory`: active for-sale listing inventory for the month.
- `new_listings`: listings that came on market during the month.
- `median_list_price`: median listed price in USD.
