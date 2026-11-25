# Installation

The package is a Rust library with Python bindings. Releases on PyPI and
crates.io are planned; until then, install from source.

## Python

Building needs a Rust toolchain ([rustup.rs](https://rustup.rs)) and Python
3.10 or later:

```bash
git clone https://github.com/njneeteson/cortical-bone-mapping-rs
cd cortical-bone-mapping-rs
pip install ./python
```

This compiles the extension in release mode, which takes a few minutes the
first time. The only runtime dependency is NumPy. Check the install:

```bash
python -c "import cortical_bone_mapping_rs as cbm; print(cbm.__version__)"
```

Optional extras:

| extra | for |
|---|---|
| `pip install ./python[test]` | the test suite (`pytest python/tests`) |
| `pip install ./python[docs]` | the figures and this documentation |

For development, build in place with [maturin](https://www.maturin.rs):

```bash
cd python
pip install maturin
maturin develop --release
```

## Rust

Add the crate from the repository:

```toml
[dependencies]
cortical-bone-mapping = { git = "https://github.com/njneeteson/cortical-bone-mapping-rs" }
```

See [Using the Rust crate](rust.md).

## Threads

Fits use all cores by default. Set `threads=` on any call (or in
{class}`~cortical_bone_mapping_rs.Options`) to limit them; results are
bitwise identical whatever the number of threads.
