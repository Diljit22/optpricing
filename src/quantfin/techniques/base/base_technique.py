from abc import ABC, abstractmethod

from quantfin.techniques.base.pricing_result import PricingResult
from quantfin.atoms.rate                import Rate
from quantfin.atoms.stock               import Stock
from quantfin.atoms.option              import Option
from quantfin.models.base               import BaseModel

class BaseTechnique(ABC):
    """
    Accept the core objects, return price (optionally Greeks/IV).
    """
    @abstractmethod
    def price(
        self,
        option: Option,
        stock:  Stock,
        model:  BaseModel,
        rate:   Rate
    ) -> PricingResult:
        ...