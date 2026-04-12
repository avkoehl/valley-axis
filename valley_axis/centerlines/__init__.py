import geopandas as gpd
import xarray as xr

from .glacial import build_cost_graph, route_segments
from .skeleton import (
    raster_skeletonization,
    raster_skeleton_to_graph,
    filter_inlets,
    mask_constrained_snap,
    route_paths,
)
from .topology import calculate_hierarchy
from .conversions import build_gdf, build_raster
from .flowline_endpoints import identify_network_pixels
from .segmentation import compute_path_segmentation
from ..inputs import align_inputs


def get_centerlines(
    dem: xr.DataArray,
    region_raster: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    method: str = "mcp",
    **kwargs,
) -> tuple[gpd.GeoDataFrame, xr.DataArray, xr.DataArray]:
    if method not in ("mcp", "skeleton"):
        raise ValueError(f"method must be 'mcp' or 'skeleton', got '{method}'")

    pixel_size = float(abs(dem.rio.resolution()[0]))
    dem, region_raster, flowlines = align_inputs(dem, region_raster, flowlines)
    mask_array = region_raster.values

    networks_dict = identify_network_pixels(flowlines, region_raster)

    # 1. Route raw pixel segments
    raw_segments_dict = {}
    if method == "mcp":
        mcp = build_cost_graph(dem.values, mask_array, pixel_size, **kwargs)
        for net_id, (inlets, outlet) in networks_dict.items():
            raw_segments_dict[net_id] = route_segments(mcp, inlets, outlet)
    else:
        skeleton_array = raster_skeletonization(mask_array, **kwargs)
        graph = raster_skeleton_to_graph(skeleton_array)
        for net_id, (inlets, outlet) in networks_dict.items():
            filtered_inlets = filter_inlets(inlets, mask_array, pixel_size, **kwargs)
            snapped_inlets, snapped_outlet = mask_constrained_snap(
                filtered_inlets, outlet, skeleton_array, mask_array
            )
            raw_segments_dict[net_id] = route_paths(
                graph, snapped_inlets, snapped_outlet
            )

    # 2. Calculate topology (Strahler order + path labels) per network
    labeled_segments = {
        net_id: calculate_hierarchy(segs) for net_id, segs in raw_segments_dict.items()
    }

    # 3. Build outputs
    centerlines_gdf = build_gdf(labeled_segments, dem.rio.transform(), dem.rio.crs)
    centerlines_raster = build_raster(labeled_segments, region_raster)

    # 4. Compute path segmentation map
    segment_to_path = dict(
        zip(centerlines_gdf["segment_id"], centerlines_gdf["path_label"])
    )
    path_map_array = compute_path_segmentation(
        centerlines_raster.values, mask_array, segment_to_path
    )
    path_map = xr.DataArray(
        path_map_array, coords=region_raster.coords, dims=region_raster.dims
    )
    path_map = path_map.rio.write_crs(region_raster.rio.crs)

    return centerlines_gdf, centerlines_raster, path_map
