"""
Validate inputs for cli programs (elbow.py and cli.py).
"""

import argparse

MAX_K = 256


def _bounded_numeric(min_value=None, min_exclusive=False, max_value=None):
    """Build an argparse type callback which by using caster checks correct bounds."""

    def _validate(value: str):
        result = int(value)
        if min_value is not None:
            if min_exclusive and result <= min_value:
                raise argparse.ArgumentTypeError(f"must be > {min_value}, got {result}")
            if not min_exclusive and result < min_value:
                raise argparse.ArgumentTypeError(
                    f"must be >= {min_value}, got {result}"
                )
        if max_value is not None and result > max_value:
            raise argparse.ArgumentTypeError(f"must be <= {max_value}, got {result}")
        return result

    return _validate


positive_capped_int = _bounded_numeric(min_value=1, max_value=MAX_K)

positive_int = _bounded_numeric(min_value=1)

non_negative_int = _bounded_numeric(min_value=0)

positive_float = _bounded_numeric(min_value=0, min_exclusive=True)

elbow_max_k_int = _bounded_numeric(min_value=3, max_value=MAX_K)
