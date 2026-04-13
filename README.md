# csc6903-project

This repository has submodules, so clone with:
```bash
git clone --recurse-submodules https://github.com/samhaswon/csc6903-project
```

## Requirements

### General

[Python](https://www.python.org/downloads/) (~3.10+)

- Dependencies are in `./requirements.txt`, so:
    ```bash
    python3 -m pip install -r ./requirements.txt
    ```

    - If you do not have an Nvidia GPU, there are some notes about alternative packages you may wish to install instead.

    - If you have an Nvidia GPU, you may wish to install the optional dependency listed at the bottom of the requirements file.

`gcc` or the equivalent for your platform

### Zero-knowledge proof

[Rust](https://rust-lang.org/tools/install/)

[Node.js](https://nodejs.org/en/download/current)

- `npm install -g snarkjs`

To install the ZKP dependency, Circom 2, see: https://docs.circom.io/getting-started/installation

## Assorted Notes

EIA LCOE (2020)

- https://en.wikipedia.org/wiki/Cost_of_electricity_by_source#Bank_of_America_(2023)

| Generation source                | USD$/MWh |
| :------------------------------- | :------: |
| Biomass                          |    95    |
| Coal (ultra-supercritical)       |    76    |
| Natural Gas (combined cycle)     |    38    |
| Natural Gas (combustion turbine) |    67    |
| Nuclear                          |    82    |
| Solar                            |    36    |
| Wind                             |    40    |





