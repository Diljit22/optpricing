from typing import Dict

class ParamValidator:
    """Utility for validating model parameters."""

    @staticmethod
    def require(
        params: Dict[str, float],
        required: list[str],
        *,
        model: str
    ) -> None:
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"{model}: missing required parameters: {', '.join(missing)}")

    @staticmethod
    def positive(
        params: Dict[str, float],
        keys: list[str],
        *,
        model: str
    ) -> None:
        nonpos = [k for k in keys if params.get(k, 0.0) <= 0.0]
        if nonpos:
            raise ValueError(f"{model}: parameters must be positive: {', '.join(nonpos)}")

    @staticmethod
    def bounded(
        params: Dict[str, float],
        key: str,
        low: float,
        high: float,
        *,
        model: str
    ) -> None:
        val = params.get(key)
        if val is None or not (low <= val <= high):
            raise ValueError(f"{model}: parameter '{key}' must be in [{low}, {high}], got {val}")
