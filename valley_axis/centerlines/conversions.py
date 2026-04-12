import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import rasterio
import xarray as xr
import networkx as nx


def lines_to_network(lines):
    """Converts a GeoDataFrame of LineStrings into a directed NetworkX graph."""
    G = nx.DiGraph()
    for index, line in lines.iterrows():
        start = line.geometry.coords[0]
        end = line.geometry.coords[-1]
        G.add_edge(start, end, streamID=index)
    return G


def build_gdf(labeled_segments, transform, crs) -> gpd.GeoDataFrame:
    """Converts hierarchically labeled pixel reaches into a vector GeoDataFrame."""
    features = []
    segment_id = 1

    for net_id, feature_list in labeled_segments.items():
        for feature in feature_list:
            segment_nodes = feature["nodes"]
            if len(segment_nodes) < 2:
                continue

            rows = [p[0] for p in segment_nodes]
            cols = [p[1] for p in segment_nodes]
            xs, ys = rasterio.transform.xy(transform, rows, cols)

            features.append(
                {
                    "segment_id": segment_id,
                    "network_id": net_id,
                    "path_label": feature["path_label"],
                    "strahler_order": feature["strahler_order"],
                    "geometry": LineString(zip(xs, ys)),
                }
            )
            segment_id += 1

    if not features:
        return gpd.GeoDataFrame(
            columns=[
                "segment_id",
                "network_id",
                "path_label",
                "strahler_order",
                "geometry",
            ],
            geometry="geometry",
            crs=crs,
        )

    return gpd.GeoDataFrame(features, crs=crs)


def build_raster(labeled_segments, region_raster: xr.DataArray) -> xr.DataArray:
    """Burns the centerlines back into a raster array using segment_id."""
    out_shape = region_raster.shape
    # Use uint32 in case you have more than 65,535 segments!
    centerline_array = np.zeros(out_shape, dtype=np.uint32)

    segment_id = 1
    for net_id, feature_list in labeled_segments.items():
        for feature in feature_list:
            for r, c in feature["nodes"]:
                if 0 <= r < out_shape[0] and 0 <= c < out_shape[1]:
                    # At junctions, the later segment will overwrite the pixel.
                    # This is fine since it's just the exact junction pixel.
                    centerline_array[r, c] = segment_id

            segment_id += 1

    centerlines_da = xr.DataArray(
        centerline_array,
        coords=region_raster.coords,
        dims=region_raster.dims,
        attrs=region_raster.attrs,
    )
    centerlines_da.rio.write_crs(region_raster.rio.crs, inplace=True)
    centerlines_da.rio.write_transform(region_raster.rio.transform(), inplace=True)

    return centerlines_da
