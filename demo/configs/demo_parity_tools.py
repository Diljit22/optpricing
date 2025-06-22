from quantfin.parity.implied_rate import ImpliedRateModel
from quantfin.parity.parity_model import ParityModel


def run_parity_demonstration():
    """
    Demonstrates the use of ParityModel and ImpliedRateModel.
    """
    print("=" * 60)
    print("DEMONSTRATION: Parity and Implied Rate Tools")
    print("=" * 60)

    S0, K, T = 100.0, 100.0, 1.0
    q, r_known = 0.01, 0.03
    call_price, put_price = 8.8273, 6.8669

    # 1. Test ParityModel ---
    parity_model = ParityModel(params={})
    print("\n1. Put-Call Parity Calculations:")

    # Use the public `closed_form` method, not the internal `_closed_form_impl`
    implied_put = parity_model.closed_form(
        spot=S0, strike=K, r=r_known, t=T, q=q, call=True, option_price=call_price
    )
    implied_call = parity_model.closed_form(
        spot=S0, strike=K, r=r_known, t=T, q=q, call=False, option_price=put_price
    )

    print(
        f"  - Given Call Price {call_price:.4f} -> Implied Put Price: {implied_put:.4f} (Actual: {put_price:.4f})"
    )
    print(
        f"  - Given Put Price  {put_price:.4f} -> Implied Call Price: {implied_call:.4f} (Actual: {call_price:.4f})"
    )

    # Test ImpliedRateModel
    implied_rate_model = ImpliedRateModel(params={"eps": 1e-8, "max_iter": 100})
    print("\n2. Implied Rate Calculation:")

    implied_r = implied_rate_model.closed_form(
        call_price=call_price, put_price=put_price, spot=S0, strike=K, t=T, q=q
    )

    print(f"  - From C={call_price:.4f} and P={put_price:.4f}")
    print(f"  - Implied Rate: {implied_r:.6f} (Known Rate: {r_known:.6f})")
    print(f"  - Error:        {implied_r - r_known:+.2e}")
    print("-" * 60)


if __name__ == "__main__":
    run_parity_demonstration()
