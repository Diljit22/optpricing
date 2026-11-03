"""Small helpers shared by the correctness tests."""

from __future__ import annotations

from optpricing import Option, OptionType

# Whole-day year fractions used by the QuantLib references (Actual/365 Fixed).
HALF_YEAR = 183 / 365
THREE_QUARTERS = 274 / 365


def make_option(strike: float, maturity: float, call: bool = True) -> Option:
    return Option(
        strike=strike,
        maturity=maturity,
        option_type=OptionType.CALL if call else OptionType.PUT,
    )
