from typing import Union, Optional, Tuple, Any
import numpy as np


def round_numbers(numbers_to_round: Union[float, tuple[float], list[float]], precisions: Union[int, list[float]] = 2) -> tuple[
    float, ...]:
    """
    Round a list of numbers to a common precision, or a list of precisions in the same order
    Args:
        numbers_to_round: list of floats to round
        precisions: Either a single precision level, or a list of precisions for each rounding step

    Returns:
        Tuple of rounded numbers: num1, num2, num3 = round_numbers([num1, num2, num3])
    """
    numbers_to_round, precisions = _rounding_sanitize_inputs(numbers_to_round, precisions)
    numbers_rounded = list()
    for number, precision in zip(numbers_to_round, precisions):
        numbers_rounded += [(round(number, precision))]
    return tuple(numbers_rounded)


def scientific_notation(numbers_to_round: Union[float, tuple[float], list[float]], sig_figs: Union[int, list[float]] = 2) -> \
tuple[str, ...]:
    """
    Convert a list of numbers to scientific notations with a given number of significant figures. The numbers will be
    strings, so should be used only be used for logging or plotting endpoints
    Args:
        numbers_to_round: list of floats to round
        sig_figs: Either a single precision level, or a list of precisions for each rounding step e.g. 2 means report
        as 1.12e[order of magnitude]

    Returns:
        Tuple of rounded numbers: num1, num2, num3 = round_numbers([num1, num2, num3])
    """
    numbers_to_round, sig_figs = _rounding_sanitize_inputs(numbers_to_round, sig_figs)
    numbers_rounded = list()
    for number, precision in zip(numbers_to_round, sig_figs):
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
