from __future__ import annotations

# Core interface
from .base_config import BenchmarkConfig

# Individual demo configs
from .demo_bates import *
from .demo_blacks_approx import *
from .demo_bsm import *
from .demo_cev import *
from .demo_cgmy import *
from .demo_dupire import *
from .demo_heston import *
from .demo_hyperbolic import *
from .demo_kou import *
from .demo_merton_jump import *
from .demo_nig import *
from .demo_parity_tools import *
from .demo_perpetual_put import *
from .demo_sabr import *
from .demo_sabr_jump import *
from .demo_vg import *

__all__ = [
    "BenchmarkConfig",
    # plus whatever names each demo_*.py defines (they’ll be imported via the * imports)
]
