from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

__doc__ = """
Defines the abstract base class for all option pricing models.
"""

CF = Callable[[np.ndarray], np.ndarray]
PDECoeffs = Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray, np.ndarray]]


class BaseModel(ABC):
    """
    Abstract base class for all financial pricing models.

    This class defines a common interface for all models, including parameter
    validation, metadata flags for supported features (e.g., characteristic
    function, SDE), and methods for creating modified model instances.

    Attributes
    ----------
    name : str
        A string identifier for the model (e.g., "Black-Scholes-Merton").
    params : dict[str, float]
        A dictionary holding the model's parameters.
    supports_cf : bool
        Flag indicating if the model implements a characteristic function.
    supports_sde : bool
        Flag indicating if the model implements an SDE simulation path.
    supports_pde : bool
        Flag indicating if the model provides PDE coefficients.
    has_closed_form : bool
        Flag indicating if a closed-form solution is available.
    has_variance_process : bool
        Flag for stochastic volatility models (e.g., Heston, SABR).
    is_pure_levy : bool
        Flag for pure Levy models where the terminal value can be sampled directly.
    has_jumps : bool
        Flag for models that include a jump component.
    """

    name: str = "BaseModel"
    params: dict[str, float]
    supports_cf: bool = False
    supports_sde: bool = False
    supports_pde: bool = False
    has_closed_form: bool = False
    has_variance_process: bool = False
    is_pure_levy: bool = False
    has_jumps: bool = False

    # TODO: Rename cf_kwargs to something more specific like `pricing_args`
    # as it's used by more than just the characteristic function.
    cf_kwargs: tuple[str, ...] = ()

    __slots__ = ("params",)

    def __init__(self, params: dict[str, float]) -> None:
        """
        Initializes the model and validates its parameters.

        Parameters
        ----------
        params : dict[str, float]
            A dictionary of parameter names to values for the model.
        """
        self.params = params
        self._validate_params()

    @abstractmethod
    def _validate_params(self) -> None:
        """
        Abstract method for subclasses to validate their specific parameters.

        This method should be implemented by all concrete model classes to ensure
        that the provided `self.params` dictionary contains all required keys
        and that their values are valid.
        """
        raise NotImplementedError

    def cf(self, **kwargs: Any) -> CF:
        """
        Return the characteristic function of the log-price process.

        Raises
        ------
        NotImplementedError
            If the model does not support a characteristic function.
        """
        if not self.supports_cf:
            raise NotImplementedError(
                f"{self.name} does not support characteristic functions."
            )
        return self._cf_impl(**kwargs)

    @abstractmethod
    def _cf_impl(self, **kwargs: Any) -> CF:
        """Internal implementation of the characteristic function."""
        ...

    def get_sde_sampler(self, **kwargs: Any) -> Callable:
        """
        Return a function that can be used to sample paths from the model's SDE.

        Raises
        ------
        NotImplementedError
            If the model does not support SDE sampling.
        """
        if not self.supports_sde:
            raise NotImplementedError(f"{self.name} does not support SDE sampling.")
        return self._sde_impl(**kwargs)

    @abstractmethod
    def _sde_impl(self, **kwargs: Any) -> Callable:
        """Internal implementation for returning an SDE sampler."""
        ...

    def get_pde_coeffs(self, **kwargs: Any) -> PDECoeffs:
        """
        Return the coefficients for the pricing PDE.

        Raises
        ------
        NotImplementedError
            If the model does not support PDE solving.
        """
        if not self.supports_pde:
            raise NotImplementedError(f"{self.name} does not support PDE solving.")
        return self._pde_impl(**kwargs)

    @abstractmethod
    def _pde_impl(self, **kwargs: Any) -> PDECoeffs:
        """Internal implementation for returning PDE coefficients."""
        ...

    def price_closed_form(self, *args, **kwargs) -> float:
        """
        Compute the option price using a closed-form solution, if available.

        Raises
        ------
        NotImplementedError
            If the model does not have a closed-form solution.
        """
        if not self.has_closed_form:
            raise NotImplementedError(
                f"{self.name} does not have a closed-form solution."
            )
        return self._closed_form_impl(*args, **kwargs)

    @abstractmethod
    def _closed_form_impl(self, *args, **kwargs) -> float:
        """Internal implementation of the closed-form pricing formula."""
        ...

    def with_params(self, **updated_params: float) -> BaseModel:
        """
        Create a new model instance with updated parameters.

        This is useful for calibration and sensitivity analysis.

        Parameters
        ----------
        **updated_params : float
            Keyword arguments for the parameters to update.

        Returns
        -------
        BaseModel
            A new instance of the model with the updated parameters.
        """
        new_params = {**self.params, **updated_params}
        return self.__class__(params=new_params)

    def __repr__(self) -> str:
        """Provide a formal string representation of the model."""
        param_str = ", ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"

    def __eq__(self, other: object) -> bool:
        """
        Check for equality based on class type and parameters.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        """
        Provide a hash for the model, making it usable in sets and dict keys.
        """
        return hash((self.__class__, tuple(sorted(self.params.items()))))
