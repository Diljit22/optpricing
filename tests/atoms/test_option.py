import pytest

from optpricing.atoms import ExerciseStyle, Option, OptionType


def test_option_creation():
    """
    Tests successful creation of an Option instance.
    """
    option = Option(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
    )
    assert option.strike == 100
    assert option.maturity == 1.0
    assert option.option_type == OptionType.CALL
    assert option.exercise_style == ExerciseStyle.EUROPEAN


def test_option_default_exercise_style():
    """
    Tests that the default exercise style is European.
    """
    option = Option(strike=100, maturity=1.0, option_type=OptionType.PUT)
    assert option.exercise_style == ExerciseStyle.EUROPEAN


@pytest.mark.parametrize("invalid_strike", [0, -100])
def test_option_invalid_strike(invalid_strike):
    """
    Tests that creating an option with a non-positive strike raises ValueError.
    """
    with pytest.raises(ValueError, match="Strike must be positive"):
        Option(strike=invalid_strike, maturity=1.0, option_type=OptionType.CALL)


@pytest.mark.parametrize("invalid_maturity", [0, -1])
def test_option_invalid_maturity(invalid_maturity):
    """
    Tests that creating an option with a non-positive maturity raises ValueError.
    """
    with pytest.raises(ValueError, match="Maturity must be positive"):
        Option(strike=100, maturity=invalid_maturity, option_type=OptionType.CALL)


def test_option_parity_counterpart():
    """
    Tests the parity_counterpart method for both call and put options.
    """
    call_option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
    put_option = call_option.parity_counterpart()

    assert put_option.option_type == OptionType.PUT
    assert put_option.strike == call_option.strike
    assert put_option.maturity == call_option.maturity

    call_again = put_option.parity_counterpart()
    assert call_again.option_type == OptionType.CALL
    assert call_again == call_option
