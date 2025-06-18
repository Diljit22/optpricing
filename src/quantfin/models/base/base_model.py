from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Set, Tuple
import numpy as np

CF = Callable[[complex], complex]
PDECoeffs = Callable[[np.ndarray, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]

class BaseModel(ABC):
    """
    Abstract base class for all pricing models.

    Attributes:
        name: Identifier for the model class.
        supports_cf: Whether the model provides a characteristic function.
        supports_sde: Whether the model provides an SDE sampler.
        supports_pde: Whether the model provides a PDE solver.
        has_closed_form: Whether the model has a closed-form pricing method.
        supported_lattices: Set of lattice names this model can use.
    """
    name: str = "BaseModel"
    supports_cf: bool = False
    supports_sde: bool = False
    supports_pde: bool = False
    has_closed_form: bool = False
    supported_lattices: Set[str] = set()
    cf_kwargs: Tuple[str, ...] = ("spot", "strike", "r", "q", "t", "call")
    has_variance_process: bool = False # For Heston, Bates, SABR, ...
    is_pure_levy: bool = False #For VG, NIG, CGMY, ... (terminal sampling)
    has_jumps: bool = False # For Merton, Kou, Bates

    def __init__(self, params: Dict[str, float]) -> None:
        """
        Initialize with a dict of parameter names to values.
        Validates parameters via `_validate_params`.
        """
        self.params = params
        self._validate_params()

    @abstractmethod
    def _validate_params(self) -> None:
        """Ensure required parameters are present and valid."""
        raise NotImplementedError

    def cf(self, **kwargs: Any) -> CF:
        """
        Public-facing method to get the characteristic function.
        It accepts arbitrary keyword arguments and passes them to the implementation.
        """
        if not self.supports_cf:
            raise NotImplementedError(f"{self.name} does not support characteristic functions.")
        # Pass all provided keyword arguments directly to the implementation
        return self._cf_impl(**kwargs)

    @abstractmethod
    def _cf_impl(self, **kwargs: Any) -> CF:
        """
        Internal implementation of the characteristic function.
        Must be able to handle all necessary keyword arguments.
        """
        ...

    def sde(self) -> Any:
        if not self.supports_sde:
            raise NotImplementedError(f"{self.name} does not support SDE sampling.")
        return self._sde_impl()

    @abstractmethod
    def _sde_impl(self) -> Any:
        """Return an SDE sampler object or interface for the model."""
        ...

    def pde(self) -> Any:
        if not self.supports_pde:
            raise NotImplementedError(f"{self.name} does not support PDE solving.")
        return self._pde_impl()

    @abstractmethod
    def _pde_impl(self) -> Any:
        """Return a PDE solver interface for the model."""
        ...

    def closed_form(self, *args, **kwargs) -> Any:
        if not self.has_closed_form:
            raise NotImplementedError(f"{self.name} does not have a closed-form solution.")
        return self._closed_form_impl(*args, **kwargs)

    @abstractmethod
    def _closed_form_impl(self, *args, **kwargs) -> Any:
        """Compute option price in closed-form, if available."""
        ...

    def with_params(self, **upd: float) -> BaseModel:
        """Return a new instance with updated parameters."""
        new_params = {**self.params, **upd}
        return self.__class__(params=new_params)

    def __hashable_state__(self) -> tuple[tuple[str, float], ...]:
        """Provide a hashable representation for caching or dict keys."""
        return tuple(sorted(self.params.items()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"