from optpricing.atoms import Rate


def test_rate_with_float():
    """
    Tests Rate class with a constant float rate.
    """
    rate = Rate(rate=0.05)
    assert rate.get_rate() == 0.05
    assert rate.get_rate(t=10) == 0.05  # Should be constant


def test_rate_with_callable():
    """
    Tests Rate class with a callable term structure.
    """

    def term_structure(t):
        """A simple linear term structure for testing."""
        return 0.02 + 0.01 * t

    rate = Rate(rate=term_structure)
    assert rate.get_rate(t=0) == 0.02
    assert rate.get_rate(t=1) == 0.03
    assert rate.get_rate(t=5) == 0.07
