"""
Common definitions.
"""


__all__ = [
    'celsius_to_kelvin',
    'kelvin_to_celsius'
]


KELVIN_OFFSET = 273.15


def celsius_to_kelvin(C):
    return C + KELVIN_OFFSET


def kelvin_to_celsius(K):
    return K - KELVIN_OFFSET
