from typing import Union, Optional, Tuple, Any
import numpy as np


def round_numbers(numbers_to_round: Union[float, tuple[float], list[float]], precisions: Union[int, list[float]] = 2) -> tuple[
    float, ...]:
    numbers_to_round, precisions = _rounding_sanitize_inputs(numbers_to_round, precisions)
    numbers_rounded = list()
    for number, precision in zip(numbers_to_round, precisions):
        numbers_rounded += [(round(number, precision))]
    return tuple(numbers_rounded)


def scientific_notation(numbers_to_round: Union[float, tuple[float], list[float]], precisions: Union[int, list[float]] = 2) -> \
tuple[str, ...]:
    numbers_to_round, precisions = _rounding_sanitize_inputs(numbers_to_round, precisions)
    numbers_rounded = list()
    for number, precision in zip(numbers_to_round, precisions):
        numbers_rounded += [f"{number:.{precision}e}"]
    return tuple(numbers_rounded)


def _rounding_sanitize_inputs(numbers_to_round, precisions):
    if not isinstance(numbers_to_round, list):
        numbers_to_round = list(numbers_to_round)
    if isinstance(precisions, int):
        precisions = [precisions] * len(numbers_to_round)
    elif isinstance(precisions, list) and len(precisions) != len(numbers_to_round):
        raise ValueError("If precisions are specified as a list, they must have the same length as numbers to "
                         "round.")
    return numbers_to_round, precisions
