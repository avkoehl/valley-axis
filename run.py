import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray

# from valley_axis import derive_axis
from valley_axis.interface import _glacier_centerline, _raster_skeleton
from valley_axis.sample_data import get_sample_data
from valley_axis.inputs import identify_network_pixels, align_inputs
from valley_axis.conversions import (
    segments_to_gdf,
    segments_to_raster,
)
from valley_axis.widths import compute_widths_idw, compute_widths_voronoi


data = get_sample_data()
dem = rioxarray.open_rasterio(data["dem"]).squeeze()
region_raster = rioxarray.open_rasterio(data["region"]).squeeze()
flowlines = gpd.read_file(data["flowlines"])
dem, region_raster, flowlines = align_inputs(dem, region_raster, flowlines)

networks_dict = identify_network_pixels(flowlines, region_raster)
pixel_size = dem.rio.resolution()[0]

# all_segments = _glacier_centerline(
#     dem.values, region_raster.values, networks_dict, pixel_size
# )
all_segments = _raster_skeleton(region_raster.values, networks_dict, pixel_size)

centerlines_gdf = segments_to_gdf(all_segments, dem.rio.transform(), dem.rio.crs)
centerlines_raster = segments_to_raster(all_segments, region_raster)

# widths = compute_widths_idw(centerlines_raster.values, region_raster.values, pixel_size)
widths = compute_widths_voronoi(
    centerlines_raster.values, region_raster.values, pixel_size
)
widths = xr.DataArray(
    widths,
    coords=region_raster.coords,
    dims=region_raster.dims,
    name="valley_width",
)
widths.rio.to_raster("widths_two.tif")

centerlines_gdf.to_file("new_c.gpkg", driver="GPKG")
