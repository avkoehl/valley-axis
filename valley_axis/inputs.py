from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union


def align_inputs(dem, region, flowlines):
    dem = dem.rio.reproject_match(region)
    flowlines = flowlines.to_crs(dem.rio.crs)
    valid_mask = (dem != dem.rio.nodata) & (region != region.rio.nodata)
    dem = dem.where(valid_mask, dem.rio.nodata)
    region = region.where(valid_mask, region.rio.nodata)

    polygons = [
        shape(geom)
        for geom, val in shapes(
            valid_mask.values.astype("uint8"),
            transform=region.rio.transform(),
            connectivity=8,
        )
        if val == 1
    ]
    valid_geom = unary_union(polygons)
    flowlines = (
        flowlines.clip(valid_geom).explode(index_parts=False).reset_index(drop=True)
    )
    return dem, region, flowlines
