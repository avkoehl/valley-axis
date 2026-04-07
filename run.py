import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray

# from valley_axis import derive_axis
from valley_axis.sample_data import get_sample_data
from valley_axis.inputs import identify_network_pixels
from valley_axis.centerlines import build_cost_graph, route_segments
from valley_axis.conversions import segments_to_gdf, segments_to_raster
from valley_axis.widths import compute_widths


data = get_sample_data()
dem = rioxarray.open_rasterio(data["dem"]).squeeze()
region_raster = rioxarray.open_rasterio(data["region"]).squeeze()
flowlines = gpd.read_file(data["flowlines"])

dem = dem.rio.reproject_match(region_raster)
networks_dict = identify_network_pixels(flowlines, region_raster)

pixel_size = dem.rio.resolution()[0]
mcp = build_cost_graph(dem.values, region_raster.values, pixel_size)
all_segments = {}
# ISSUE with network 2, no minimum cost path found, likely due to the fact that the outlet is not connected to the inlets in the cost graph, need to visually inspect
for net_id, (inlets, outlet) in networks_dict.items():
    segments = route_segments(mcp, inlets, outlet)
    all_segments[net_id] = segments

centerlines_gdf = segments_to_gdf(all_segments, dem.rio.transform(), dem.rio.crs)
centerlines_raster = segments_to_raster(all_segments, region_raster)

widths = compute_widths(centerlines_raster.values, region_raster.values, pixel_size)
widths = xr.DataArray(
    widths,
    coords=region_raster.coords,
    dims=region_raster.dims,
    name="valley_width",
)
