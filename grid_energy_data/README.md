# Grid Energy Data

Sources:

- Demand: https://www.neso.energy/data-portal/historic-demand-data?page=0
    - Download all files into `grid_energy_data/dataset/demand`
- Generation mix: https://www.neso.energy/data-portal/historic-generation-mix
    - Download into `grid_energy_data/dataset/`
- System frequency: https://www.neso.energy/data-portal/system-frequency-data
    - Download all files into `grid_energy_data/dataset/frequency`

## Commands

(with `venv`)

Normalize grid data:

```bash
python3 grid_energy_data/normalize_grid_dataset.py \
  --dataset-root grid_energy_data/dataset \
  --output-dir grid_energy_data/normalized \
  --workers 16 \
  --gzip-compresslevel 9 \
  --overwrite
```

Train models:
```bash
python3 grid_energy_data/test_frequency_models.py \
  --normalized-dir grid_energy_data/normalized \
  --best-summary dataset_exploration/artifacts/grid_search_summary.json \
  --output grid_energy_data/artifacts/frequency_model_test_results.json \
  --count-workers 16 \
  --materialize-workers 16 \
  --max-rows 5000000
```

## Dataset Layout

Data files are under `dataset/` and split into:

- `dataset/demand/`: annual demand files (`demanddata_YYYY.csv`, 2001 to 2025).
- `dataset/df_fuel_ckan.csv`: generation mix time series (single file).
- `dataset/frequency/`: monthly frequency files (2014 to 2026) in a mix of `.csv` and `.zip`.

## Demand Dataset

- File pattern: `dataset/demand/demanddata_YYYY.csv`
- Coverage: 2001 to 2025
- Time granularity: settlement periods (typically 1 to 48 per day, half-hourly)
- Core date/time fields:
  - `SETTLEMENT_DATE`
  - `SETTLEMENT_PERIOD`

Schema is not uniform across all years. Three variants are present:

1. 22 columns (`2001-2008`, `2023-2025`)
   - Includes all fields below, including `SCOTTISH_TRANSFER` and newer interconnector flows.
2. 17 columns (`2009-2018`)
   - Does not include `SCOTTISH_TRANSFER`, `NSL_FLOW`, `ELECLINK_FLOW`, `VIKING_FLOW`,
     `GREENLINK_FLOW`.
3. 21 columns (`2019-2022`)
   - Includes newer interconnector flows but does not include `SCOTTISH_TRANSFER`.

Full superset of observed demand columns:

- `SETTLEMENT_DATE`: settlement day.
- `SETTLEMENT_PERIOD`: half-hour index within day.
- `ND`: national demand.
- `TSD`: transmission system demand.
- `ENGLAND_WALES_DEMAND`: England and Wales demand.
- `EMBEDDED_WIND_GENERATION`: embedded wind generation.
- `EMBEDDED_WIND_CAPACITY`: embedded wind capacity.
- `EMBEDDED_SOLAR_GENERATION`: embedded solar generation.
- `EMBEDDED_SOLAR_CAPACITY`: embedded solar capacity.
- `NON_BM_STOR`: non-BM storage.
- `PUMP_STORAGE_PUMPING`: pumped storage pumping.
- `SCOTTISH_TRANSFER`: transfer from Scotland (present in some schema versions only).
- `IFA_FLOW`, `IFA2_FLOW`, `BRITNED_FLOW`, `MOYLE_FLOW`, `EAST_WEST_FLOW`, `NEMO_FLOW`:
  interconnector flows.
- `NSL_FLOW`, `ELECLINK_FLOW`, `VIKING_FLOW`, `GREENLINK_FLOW`:
  additional interconnector flows in newer schema versions.

Note: historical rows may contain `NA` placeholders in early years.

## Generation Mix Dataset

- File: `dataset/df_fuel_ckan.csv`
- Coverage: `2009-01-01T00:00:00` to `2026-04-05T23:30:00`
- Time granularity: 30 minutes
- Column count: 34

Columns:

- `DATETIME`
- Absolute generation/output fields:
  - `GAS`, `COAL`, `NUCLEAR`, `WIND`, `WIND_EMB`, `HYDRO`, `IMPORTS`, `BIOMASS`, `OTHER`,
    `SOLAR`, `STORAGE`, `GENERATION`
- Carbon/aggregate fields:
  - `CARBON_INTENSITY`, `LOW_CARBON`, `ZERO_CARBON`, `RENEWABLE`, `FOSSIL`
- Percentage fields:
  - `GAS_perc`, `COAL_perc`, `NUCLEAR_perc`, `WIND_perc`, `WIND_EMB_perc`, `HYDRO_perc`,
    `IMPORTS_perc`, `BIOMASS_perc`, `OTHER_perc`, `SOLAR_perc`, `STORAGE_perc`,
    `GENERATION_perc`, `LOW_CARBON_perc`, `ZERO_CARBON_perc`, `RENEWABLE_perc`, `FOSSIL_perc`

## Frequency Dataset

- Folder: `dataset/frequency/`
- Coverage: 2014 to 2026
- File count: 146 monthly files
  - 107 `.csv`
  - 39 `.zip` (zips contain CSV data)
- Stable schema across all files: 2 columns
  - `dtm`: timestamp
  - `f`: system frequency (Hz)

Observed `dtm` timestamp formats vary by period:

- `DD/MM/YYYY HH:MM:SS` (example: `01/01/2014 00:00:00`)
- `YYYY-MM-DD HH:MM:SS +0000` (example: `2019-01-01 00:00:00 +0000`)
- `YYYY-MM-DD HH:MM:SS` (example: `2020-01-01 00:00:00`)

When combining frequency files, normalize `dtm` to a single timezone-aware format before
analysis.
