from abc import ABC, abstractmethod

from techniques.base.pricing_result import PricingResult
from atoms.rate import Rate
from atoms.stock import Stock
from atoms.option import Option
from models.base.base_model import BaseModel

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