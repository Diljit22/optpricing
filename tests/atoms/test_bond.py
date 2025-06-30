import pytest

from optpricing.atoms import ZeroCouponBond


def test_zerocouponbond_creation():
    """
    Tests successful creation of a ZeroCouponBond instance.
    """
    bond = ZeroCouponBond(maturity=1.0, face_value=100.0)
    assert bond.maturity == 1.0
    assert bond.face_value == 100.0


def test_zerocouponbond_default_face_value():
    """
    Tests that the default face value is correctly assigned.
    """
    bond = ZeroCouponBond(maturity=0.5)
    assert bond.face_value == 1.0


@pytest.mark.parametrize("invalid_maturity", [0, -1])
def test_zerocouponbond_invalid_maturity(invalid_maturity):
    """
    Tests that creating a bond with zero or negative maturity raises ValueError.
    """
    with pytest.raises(ValueError, match="Maturity must be positive"):
        ZeroCouponBond(maturity=invalid_maturity)
