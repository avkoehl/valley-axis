import numpy as np
import geopandas as gpd
import networkx as nx
import xarray as xr
import rasterio.transform
from shapely.geometry import Point
from skimage.morphology import remove_small_holes


def flowlines_to_endpoints(
    flowlines: gpd.GeoDataFrame,
    mask: xr.DataArray,
    search_radius: int = 3,
) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    """
    Convert flowlines into (inlets, outlet) pixel pairs, one per network.

    A "network" is a connected set of flowlines sharing a single outlet.
    Endpoints are snapped to the nearest valid mask pixel within search_radius.
    Networks whose outlet or all inlets fail to snap are dropped.
    """
    flowlines = flowlines.to_crs(mask.rio.crs)
    transform = mask.rio.transform()
    mask_array = mask.values == 1

    G = nx.DiGraph()
    for idx, row in flowlines.iterrows():
        coords = list(row.geometry.coords)
        G.add_edge(tuple(coords[0]), tuple(coords[-1]), streamID=idx)

    outlet_points = [n for n in G.nodes() if G.out_degree(n) == 0]

    networks = []
    for outlet_point in outlet_points:
        upstream = nx.ancestors(G, outlet_point) | {outlet_point}
        sub = G.subgraph(upstream)
        inflow_points = [
            n for n in sub.nodes() if sub.in_degree(n) == 0 and sub.out_degree(n) > 0
        ]

        outlet_pixel = _snap_point(outlet_point, transform, mask_array, search_radius)
        if outlet_pixel is None:
            continue

        inlet_pixels = [
            p
            for p in (
                _snap_point(pt, transform, mask_array, search_radius)
                for pt in inflow_points
            )
            if p is not None
        ]
        if not inlet_pixels:
            continue

        networks.append((inlet_pixels, outlet_pixel))

    return networks


def fill_holes(mask: xr.DataArray, max_hole_size: int = 500) -> xr.DataArray:
    """
    Fill holes smaller than max_hole_size pixels in a binary valley mask.

    Useful before calling get_centerlines to prevent the skeleton from
    wrapping around small internal holes (e.g., islands, data gaps).
    """
    filled = remove_small_holes(mask.values == 1, max_size=max_hole_size)
    out = xr.DataArray(
        filled.astype(np.uint8),
        coords=mask.coords,
        dims=mask.dims,
        attrs=mask.attrs,
    )
    # preserve NODATA value and georeferencing metadata
    nodata_value = mask.rio.nodata
    if nodata_value is not None:
        out = out.where(mask != nodata_value, other=nodata_value)
        out.rio.write_nodata(nodata_value, inplace=True)

    out.rio.write_crs(mask.rio.crs, inplace=True)
    out.rio.write_transform(mask.rio.transform(), inplace=True)
    return out


# -- internal ---------------------------------------------------------------


def _snap_point(point, transform, mask_array, search_radius):
    x, y = (point.x, point.y) if isinstance(point, Point) else point
    r, c = rasterio.transform.rowcol(transform, x, y)
    h, w = mask_array.shape

    for dr in range(-search_radius, search_radius + 1):
        for dc in range(-search_radius, search_radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask_array[nr, nc]:
                return (nr, nc)
    return None
