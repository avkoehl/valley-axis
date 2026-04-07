import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import rasterio
import networkx as nx


def lines_to_network(lines):
    """Converts a GeoDataFrame of LineStrings into a directed NetworkX graph."""
    G = nx.DiGraph()
    for index, line in lines.iterrows():
        start = line.geometry.coords[0]
        end = line.geometry.coords[-1]
        G.add_edge(start, end, streamID=index)
    return G


def segments_to_gdf(all_segments, transform, crs):
    """Converts pixel-based segments into a vector GeoDataFrame."""
    records = []
    segment_id = 1

    for net_id, segments in all_segments.items():
        for segment in segments:
            if len(segment) < 2:
                continue

            rows = [p[0] for p in segment]
            cols = [p[1] for p in segment]
            xs, ys = rasterio.transform.xy(transform, rows, cols)

            records.append(
                {
                    "segment_id": segment_id,
                    "network_id": net_id,
                    "geometry": LineString(zip(xs, ys)),
                }
            )
            segment_id += 1

    if not records:
        return gpd.GeoDataFrame(
            columns=["segment_id", "network_id", "geometry"], crs=crs
        )

    return gpd.GeoDataFrame(records, crs=crs)


def segments_to_raster(all_segments, base_raster):
    """Burns the segmented paths into a new raster using the input raster as a template."""
    raster_array = np.zeros(base_raster.shape, dtype=np.int32)
    segment_id = 1

    for net_id, segments in all_segments.items():
        for segment in segments:
            if len(segment) < 2:
                continue

            for row, col in segment:
                raster_array[row, col] = segment_id

            segment_id += 1

    return base_raster.copy(data=raster_array)
