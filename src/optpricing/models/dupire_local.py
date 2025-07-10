from __future__ import annotations

from typing import Any

from optpricing.models.base import BaseModel

__doc__ = """
Defines the Dupire Local Volatility model.
"""


class DupireLocalVolModel(BaseModel):
    """Dupire Local Volatility model."""

    name: str = "Dupire Local Volatility"
    supports_sde: bool = True
    is_local_vol: bool = True

    default_params = {"vol_surface": lambda T, K: 0.2}

    def __init__(self, params: dict[str, Any] | None = None):
        """
        Initializes the Dupire Local Volatility model.

        The key parameter for this model is 'vol_surface', a callable function
        that takes maturity (T) and strike (K) and returns the local volatility.

        Parameters
        ----------
        params : dict[str, Any] | None, optional
            A dictionary of model parameters. If None, a default constant
            volatility surface is used. Defaults to None.
        """
        super().__init__(params or self.default_params)

    def _validate_params(self) -> None:
        if "vol_surface" not in self.params or not callable(self.params["vol_surface"]):
            raise ValueError(
                "Dupire model requires a callable 'vol_surface' in its parameters."
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DupireLocalVolModel):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(sorted(self.params.items()))))

    def __repr__(self) -> str:
        """Custom representation to handle the vol_surface function."""
        # Get the name of the function or show its type
        vol_surface_repr = getattr(
            self.params["vol_surface"],
            "__name__",
            str(type(self.params["vol_surface"])),
        )
        return f"{self.__class__.__name__}(vol_surface={vol_surface_repr})"

    #  Abstract Method Implementations
    def _sde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Dupire uses a specialized kernel.")

    def _cf_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _pde_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _closed_form_impl(self, **kwargs: Any) -> Any:
        raise NotImplementedError
