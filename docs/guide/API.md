# Guide: Programmatic API

The `optpricing` Python API provides a powerful, object-oriented framework for building custom financial analysis scripts and workflows. Its modular design allows you to mix and match models, techniques, and data to suit your specific research needs.

This guide covers the fundamental components and provides examples for common use cases.

---

## Core Concepts: Atoms, Models, and Techniques

All programmatic workflows are built on three core components:

1. **Atoms**: Immutable data classes like `Option`, `Stock`, and `Rate` that represent the basic inputs.
2. **Models**: Classes representing financial theories, such as `BSMModel` or `HestonModel` (and many others).
3. **Techniques**: Classes representing numerical algorithms, such as `ClosedFormTechnique` or `MonteCarloTechnique` (and many others).

You instantiate these components and pass them to a technique's `price` method to get a result.

## Example 1: Pricing a Standard European Option

This example demonstrates the most common use case: pricing a European call option using the Black-Scholes-Merton model's analytic formula.

```python
from optpricing import Option, OptionType, Rate, Stock
from optpricing.models import BSMModel
from optpricing.techniques import ClosedFormTechnique

# 1. Define the core Atoms
option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
stock = Stock(spot=100.0, dividend=0.01)
rate = Rate(rate=0.05)

# 2. Instantiate the Model and Technique
bsm_model = BSMModel(params={"sigma": 0.20})
cf_technique = ClosedFormTechnique()

# 3. Calculate the price and Greeks
result = cf_technique.price(option, stock, bsm_model, rate)
delta = cf_technique.delta(option, stock, bsm_model, rate)
gamma = cf_technique.gamma(option, stock, bsm_model, rate)

print(f"The option price is: {result.price:.4f}")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
```

## Example 2: Pricing an American Option with Monte Carlo

The API's flexibility allows you to easily switch to more complex scenarios. Here, we price an American option using the Heston model with the Longstaff-Schwartz Monte Carlo method.

```python
from optpricing.models import HestonModel
from optpricing.techniques import AmericanMonteCarloTechnique

# 1. Define an American option Atom
american_option = Option(strike=105, maturity=1.0, option_type=OptionType.PUT)

# 2. Instantiate a more complex model and the appropriate technique
heston_model = HestonModel(params={
    "v0": 0.04,
    "kappa": 2.0,
    "theta": 0.05,
    "rho": -0.7,
    "vol_of_vol": 0.5
})
lsmc_technique = AmericanMonteCarloTechnique(n_paths=20000, n_steps=100)

# 3. Calculate the price
# Note: The 'v0' parameter is passed as a keyword argument for the simulation
result = lsmc_technique.price(
    american_option,
    stock,
    heston_model,
    rate,
    v0=heston_model.params['v0']
)
print(f"The American option price is: {result.price:.4f}")
```

## Example 3: Calculating Implied Volatility

The library includes mixins for common tasks like calculating implied volatility. This functionality is available on most technique objects.

```python
# Using the components from Example 1
target_price = 7.50

iv = cf_technique.implied_volatility(
    option,
    stock,
    bsm_model,
    rate,
    target_price=target_price
)

print(f"Implied volatility for a target price of ${target_price:.2f}: {iv:.4%}")
```

## Example 4: Pricing an Interest Rate Product

The framework is not limited to equity options. It can be used to price other derivatives, such as Zero-Coupon Bonds using short-rate models.

```python
from optpricing.atoms import ZeroCouponBond
from optpricing.models import VasicekModel

# 1. Define a ZeroCouponBond Atom
bond = ZeroCouponBond(maturity=5.0, face_value=1000.0)

# 2. For rate models, the 'stock.spot' is re-interpreted as the initial short rate r0
r0_stock = Stock(spot=0.03)
# The 'rate' atom is ignored by rate models but required by the technique signature
dummy_rate = Rate(rate=0.0)

# 3. Instantiate the Vasicek model
vasicek_model = VasicekModel(params={"kappa": 0.86, "theta": 0.05, "sigma": 0.02})

# 4. Price using the ClosedFormTechnique
price_result = cf_technique.price(bond, r0_stock, vasicek_model, dummy_rate)
print(f"Price of 5-Year ZCB under Vasicek model: ${price_result.price:.2f}")
```

## Advanced Usage: Extending the Library

The library's true power lies in its extensibility. By inheriting from `BaseModel` and `BaseTechnique`, you can easily add your own custom models and pricing algorithms, which will automatically integrate with the existing framework.

For a complete and detailed list of all available classes, methods, and functions, please consult the full **[API Reference](../reference/index.md)**.
