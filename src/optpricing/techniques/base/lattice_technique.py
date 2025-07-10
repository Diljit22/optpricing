from __future__ import annotations

from abc import abstractmethod
from typing import Any

from optpricing.atoms import Option, Rate, Stock
from optpricing.models import BaseModel

from .base_technique import BaseTechnique
from .greek_mixin import GreekMixin
from .iv_mixin import IVMixin
from .pricing_result import PricingResult

__doc__ = """
Provides a base class for lattice-based pricing techniques.
"""


class LatticeTechnique(BaseTechnique, GreekMixin, IVMixin):
    """
    Abstract base class for lattice-based pricing techniques.

    This class provides a caching mechanism for Greek calculations. The main
    pricing method, `_price_and_get_nodes`, is designed to return both the
    option price and the values of the adjacent nodes at the first and second
    time steps, which are then used for instantaneous delta and gamma calculations
    without rebuilding the tree.
    """

    def __init__(
        self,
        steps: int = 200,
        is_american: bool = False,
    ):
        """
        Initializes the lattice technique.

        Parameters
        ----------
        steps : int, optional
            The number of time steps in the lattice, by default 200.
        is_american : bool, optional
            True if pricing an American option, False for European. Defaults to False.
        """
        self.steps = int(steps)
        self.is_american = bool(is_american)
        self._cached_nodes: dict[str, Any] = {}
        self._cache_key: tuple | None = None

    def _get_cache_key(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
    ) -> tuple:
        """Creates a unique, hashable key from the pricing inputs."""
        return (option, stock, model, rate, self.steps, self.is_american)

    def price(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> PricingResult:
        """Prices the option and caches the necessary nodes for Greek calculations."""
        self._cache_key = self._get_cache_key(option, stock, model, rate)
        results = self._price_and_get_nodes(option, stock, model, rate)
        self._cached_nodes = results
        return PricingResult(price=results["price"])

    def delta(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> float:
        """Calculates delta, using cached nodes if available and valid."""
        if self._get_cache_key(option, stock, model, rate) != self._cache_key:
            self.price(option, stock, model, rate)

        cache = self._cached_nodes
        return (cache["price_up"] - cache["price_down"]) / (
            cache["spot_up"] - cache["spot_down"]
        )

    def gamma(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        **kwargs: Any,
    ) -> float:
        """Calculates gamma, using cached nodes if available and valid."""
        if self._get_cache_key(option, stock, model, rate) != self._cache_key:
            self.price(option, stock, model, rate)

        cache = self._cached_nodes
        if "price_mid" in cache:  # Trinomial Case
            h1 = cache["spot_up"] - cache["spot_mid"]
            h2 = cache["spot_mid"] - cache["spot_down"]
            term1 = (cache["price_up"] - cache["price_mid"]) / h1
            term2 = (cache["price_mid"] - cache["price_down"]) / h2
            return 2 * (term1 - term2) / (h1 + h2)
        else:  # Binomial Case
            h_up = cache["spot_uu"] - cache["spot_ud"]
            h_down = cache["spot_ud"] - cache["spot_dd"]
            delta_up = (cache["price_uu"] - cache["price_ud"]) / h_up
            delta_down = (cache["price_ud"] - cache["price_dd"]) / h_down
            avg_spot_change = 0.5 * (cache["spot_uu"] - cache["spot_dd"])
            return (delta_up - delta_down) / avg_spot_change

    @abstractmethod
    def _price_and_get_nodes(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
    ) -> dict[str, Any]:
        """
        Abstract method for concrete lattice implementations.

        This method should perform the actual lattice calculation and return a
        dictionary containing the option price and all necessary adjacent node
        values for calculating delta and gamma.
        """
        raise NotImplementedError
