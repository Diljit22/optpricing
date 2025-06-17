from dataclasses import dataclass

@dataclass(frozen=True)
class PricingResult:
    price: float
    Greeks: dict[str, float] | None = None
    implied_vol: float   | None = None