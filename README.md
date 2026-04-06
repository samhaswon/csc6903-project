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

    - If you have an Nvidia GPU, you may wish to install the optional dependency listed at the bottom of the requirements file.

`gcc` or the equivalent for your platform

### Zero-knowledge proof

[Rust](https://rust-lang.org/tools/install/)

[Node.js](https://nodejs.org/en/download/current)

- 

- `npm install -g snarkjs`

To install the ZKP dependency, see: https://docs.circom.io/getting-started/installation