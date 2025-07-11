# Welcome to optpricing

`optpricing` is a Python library for pricing and calibrating financial derivatives.
It provides:

- A rich **command-line interface** for ad-hoc pricing, calibration, backtests and more.
- An **interactive dashboard** powered by Streamlit for visual exploration.
- A comprehensive **Python API** exposing models, techniques, data-managers, workflows, and core data structures.

---

## Quickstart

Install:

```bash
pip install optpricing
```

Run the dashboard:

```bash
optpricing dashboard
```

Run the price command:

```bash
optpricing price --ticker AAPL --strike 150 --maturity 2025-12-31 --model BSM --param sigma=0.22
```

---

## API Reference

Below is the top-level API. You can scroll down to explore every subpackage, module, class and function.

[Browse the API Reference](reference/index.md)

---

## Guides

- [Introduction](guide/introduction.md)
- [Installation](guide/installation.md)
- [Getting Started](guide/getting_started.md)
- [Dashboard](guide/dashboard.md)
- [Examples](guide/examples.md)
