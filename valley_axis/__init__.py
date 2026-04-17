from dataclasses import dataclass

import xarray as xr

from .centerlines import get_centerlines, Centerlines
from .allocation import get_allocation
from .widths import get_widths
from .helpers import flowlines_to_endpoints, fill_holes


@dataclass
class ValleyResult:
    centerlines: Centerlines
    allocation: xr.DataArray
    widths: xr.DataArray


def measure_valley(
    mask: xr.DataArray,
    networks: list[tuple[list[tuple[int, int]], tuple[int, int]]],
    width_method: str = "laplace",
    inlet_distance_threshold: float = 100.0,
) -> ValleyResult:
    """
    Full pipeline: centerlines → segment allocation → widths.

    See individual functions for details on each step.
    """
    centerlines = get_centerlines(
        mask, networks, inlet_distance_threshold=inlet_distance_threshold
    )
    allocation = get_allocation(centerlines, mask)
    widths = get_widths(centerlines, mask, allocation=allocation, method=width_method)
    return ValleyResult(centerlines=centerlines, allocation=allocation, widths=widths)


__all__ = [
    "measure_valley",
    "get_centerlines",
    "get_allocation",
    "get_widths",
    "flowlines_to_endpoints",
    "fill_holes",
    "Centerlines",
    "ValleyResult",
]
