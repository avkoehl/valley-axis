import geopandas as gpd
import networkx as nx
import rasterio
from shapely.geometry import Point
import xarray as xr

from .conversions import lines_to_network


def identify_network_pixels(
    flowlines_gdf: gpd.GeoDataFrame, region_raster: xr.DataArray
) -> dict[int, tuple[list[tuple[int, int]], tuple[int, int]]]:
    """Extracts valid inlet and outlet pixel indices for each distinct river network."""
    flowlines = _assign_network_ids(flowlines_gdf.copy())
    transform = region_raster.rio.transform()
    mask_array = region_raster.values
    networks_dict = {}

    for net_id, group in flowlines.groupby("network_id"):
        inflow_points, outlet_point = _find_network_endpoints(group)

        if not outlet_point:
            continue

        outlet_pixel = _point_to_valid_pixel(outlet_point, transform, mask_array)
        if not outlet_pixel:
            continue

        inlet_pixels = []
        for pt in inflow_points:
            pixel = _point_to_valid_pixel(pt, transform, mask_array)
            if pixel:
                inlet_pixels.append(pixel)

        if inlet_pixels and outlet_pixel:
            networks_dict[net_id] = (inlet_pixels, outlet_pixel)

    return networks_dict


def _assign_network_ids(flowlines):
    """Identifies distinct river networks and assigns a unique network_id."""
    graph = lines_to_network(flowlines)
    outlets = [node for node in graph.nodes() if graph.out_degree(node) == 0]

    if len(outlets) == 1:
        flowlines["network_id"] = 1
        return flowlines

    flowlines["network_id"] = None
    for i, outlet in enumerate(outlets, start=1):
        upstream = nx.ancestors(graph, outlet)
        upstream.add(outlet)
        subgraph = graph.subgraph(upstream)
        streams = list(
            set(data["streamID"] for _, _, data in subgraph.edges(data=True))
        )
        flowlines.loc[streams, "network_id"] = i

    return flowlines


def _find_network_endpoints(single_network_gdf):
    """Finds inflow points and the outlet point for a SINGLE network."""
    startpoints = set()
    endpoints = set()

    for line in single_network_gdf.geometry:
        coords = list(line.coords)
        startpoints.add(tuple(coords[0]))
        endpoints.add(tuple(coords[-1]))

    inflows = [Point(pt) for pt in startpoints - endpoints]
    outlets = [Point(pt) for pt in endpoints - startpoints]
    outlet = outlets[0] if outlets else None

    return inflows, outlet


def _point_to_valid_pixel(point, transform, mask_array, search_radius=3):
    """Snaps a spatial point to the nearest valid pixel within a search radius."""
    r, c = rasterio.transform.rowcol(transform, point.x, point.y)
    height, width = mask_array.shape

    for dr in range(-search_radius, search_radius + 1):
        for dc in range(-search_radius, search_radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and mask_array[nr, nc] == 1:
                return (nr, nc)
    return None
