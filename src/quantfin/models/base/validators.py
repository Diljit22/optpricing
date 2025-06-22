from __future__ import annotations

__doc__ = """
Provides a utility class for validating parameters passed to financial models.
"""


class ParamValidator:
    """
    A utility class containing static methods for model parameter validation.

    This class is not meant to be instantiated. It serves as a namespace
    for common validation logic used by `BaseModel` subclasses.
    """

    @staticmethod
    def require(params: dict[str, float], required: list[str], *, model: str) -> None:
        """
        Check for the presence of required parameters.

        Parameters
        ----------
        params : dict[str, float]
            The dictionary of parameters to validate.
        required : list[str]
            A list of parameter names that must be present in `params`.
        model : str
            The name of the model performing the validation, for error messages.

        Raises
        ------
        ValueError
            If any of the required parameters are missing.
        """
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"{model}: missing required parameters: {', '.join(missing)}"
            )

    @staticmethod
    def positive(params: dict[str, float], keys: list[str], *, model: str) -> None:
        """
        Check if specified parameters are strictly positive.

        Parameters
        ----------
        params : dict[str, float]
            The dictionary of parameters to validate.
        keys : list[str]
            A list of parameter names that must be positive.
        model : str
            The name of the model performing the validation, for error messages.

        Raises
        ------
        ValueError
            If any of the specified parameters are not strictly positive.
        """
        nonpos = [k for k in keys if params.get(k, 0.0) <= 0.0]
        if nonpos:
            raise ValueError(
                f"{model}: parameters must be positive: {', '.join(nonpos)}"
            )

    @staticmethod
    def bounded(
        params: dict[str, float], key: str, low: float, high: float, *, model: str
    ) -> None:
        """
        Check if a parameter is within a specified inclusive range.

        Parameters
        ----------
        params : dict[str, float]
            The dictionary of parameters to validate.
        key : str
            The name of the parameter to check.
        low : float
            The lower bound of the valid range.
        high : float
            The upper bound of the valid range.
        model : str
            The name of the model performing the validation, for error messages.

        Raises
        ------
        ValueError
            If the parameter is outside the [low, high] range.
        """
        val = params.get(key)
        if val is None or not (low <= val <= high):
            raise ValueError(
                f"{model}: parameter '{key}' must be in [{low}, {high}], got {val}"
            )
