from __future__ import annotations

from optpricing import Option, OptionType, Rate, Stock, ZeroCouponBond
from optpricing.models import BSMModel, CIRModel, VasicekModel
from optpricing.techniques import ClosedFormTechnique

# Define an option, underlying and rate
option = Option(strike=105, maturity=1.0, option_type=OptionType.CALL)
stock = Stock(spot=100, dividend=0.01)
rate = Rate(rate=0.05)

# Choose a model and technique
bsm_model = BSMModel(params={"sigma": 0.20})
cf_technique = ClosedFormTechnique()

result = cf_technique.price(option, stock, bsm_model, rate)
print(f"The option price is: {result.price:.4f}")


delta = cf_technique.delta(option, stock, bsm_model, rate)
gamma = cf_technique.gamma(option, stock, bsm_model, rate)
vega = cf_technique.vega(option, stock, bsm_model, rate)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega:  {vega:.4f}")

target_price = 7.50
iv = cf_technique.implied_volatility(
    option, stock, bsm_model, rate, target_price=target_price
)
print(f"Implied volatility for price ${target_price:.2f}: {iv:.4%}")


# Zero Coupon Bond
bond = ZeroCouponBond(maturity=1.0)
r0_stock = Stock(spot=0.05)  # initial short rate
dummy_rate = Rate(rate=0.0)  # ignored by rate models

vasicek = VasicekModel(params={"kappa": 0.86, "theta": 0.09, "sigma": 0.02})
cir = CIRModel(params={"kappa": 0.86, "theta": 0.09, "sigma": 0.02})

p_vasi = cf_technique.price(bond, r0_stock, vasicek, dummy_rate).price
p_cir = cf_technique.price(bond, r0_stock, cir, dummy_rate).price

print(f"Vasicek ZCB Price: {p_vasi:.4f}")
print(f"CIR ZCB Price:     {p_cir:.4f}")
