# Installation

The QuantFin library is published on the Python Package Index (PyPI) and can be easily installed using `pip`. A Python version of 3.10 or higher is required.

## Standard Installation

For most use cases, including running the command-line interface for calibration and backtesting, you can install the core library with the following command:

```bash
pip install quantfin
```

This will install the library and all its core dependencies, such as numpy, scipy, and typer.

## Full Installation (with Dashboard)

The library includes an optional interactive dashboard built with Streamlit, which provides a visual way to interact with the pricing and calibration tools. To install the core library along with the dependencies needed to run the dashboard, use the [app] extra:

```bash
pip install quantfin[app]
```

This is the recommended installation if you plan to use the visual tools.

## Developer Installation

If you wish to contribute to the development of QuantFin, or if you want to make local modifications to the source code, you should clone the repository and install it in "editable" mode.

```bash
git clone https://github.com/diljit22/quantfin.git
cd quantfin
pip install -e .[app,dev]
```
