from contextlib import contextmanager

@contextmanager
def crn(rng):
    """Context-manager to save / restore RNG state (NP RNG only)."""
    state = rng.bit_generator.state
    try:
        yield
    finally:
        rng.bit_generator.state = state
