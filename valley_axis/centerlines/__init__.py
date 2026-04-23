from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import xarray as xr
import rasterio

from .derivation import derive_segments
from .annotation import annotate_segments


@dataclass
class Centerlines:
    """
    Bundles a binary centerline raster with the per-segment attributes
    DataFrame. Provides methods to produce labeled rasters on demand.
    """

    raster: xr.DataArray  # binary (uint8, 0/1)
    segments: pd.DataFrame  # per-segment attributes

    def as_gdf(self) -> gpd.GeoDataFrame:
        """
        Vectorize centerline segments into a GeoDataFrame of LineStrings.

        Geometry is derived from the `pixels` column of centerlines.segments;
        all other attribute columns are preserved. Segments with fewer than
        2 pixels are dropped.
        """
        transform = self.raster.rio.transform()
        crs = self.raster.rio.crs
        segments_df = self.segments

        geometries = []
        keep = []
        for i, pixels in enumerate(segments_df["pixels"]):
            if len(pixels) < 2:
                continue
            rows, cols = zip(*pixels)
            xs, ys = rasterio.transform.xy(transform, list(rows), list(cols))
            geometries.append(LineString(zip(xs, ys)))
            keep.append(i)

        gdf = segments_df.iloc[keep].drop(columns=["pixels"]).copy()
        gdf["geometry"] = geometries
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)

    def label_by_segment(self) -> xr.DataArray:
        """Raster labeled by segment_id. Downstream segment wins at junctions."""
        return self._burn("segment_id")

    def label_by_path(self) -> xr.DataArray:
        """Raster labeled by path_uid (globally unique across networks).

        Main stem wins at junctions.
        """
        return self._burn("path_uid")

    def _burn(self, column: str) -> xr.DataArray:
        # Descending path_label write order: highest (tributary) first, lowest
        # (main stem continuation) last — so main stem wins at junction pixels.
        # path_label is per-network; across disconnected networks pixels don't
        # overlap, so global sort on path_label is still correct.
        arr = np.zeros(self.raster.shape, dtype=np.uint32)
        if not self.segments.empty:
            ordered = self.segments.sort_values("path_label", ascending=False)
            for _, row in ordered.iterrows():
                label = int(row[column])
                for r, c in row["pixels"]:
                    if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                        arr[r, c] = label
        return _wrap_like(arr, self.raster)


def get_centerlines(
    mask: xr.DataArray,
    networks: list[tuple[list[tuple[int, int]], tuple[int, int]]],
    inlet_distance_threshold: float = 100.0,
) -> Centerlines:
    """
    Derive and annotate centerlines from a valley mask and network endpoints.

    Returns a Centerlines object holding:
      - raster: binary centerline footprint (uint8, 0/1)
      - segments: DataFrame with segment_id, network_id, path_label, path_uid,
        strahler_order, downstream_segment_id, length, pixels
    """
    pixel_size = float(abs(mask.rio.resolution()[0]))
    segments_by_network = derive_segments(
        mask, networks, inlet_distance_threshold=inlet_distance_threshold
    )
    segments_df = annotate_segments(segments_by_network, pixel_size=pixel_size)
    raster = _build_binary_raster(segments_df, mask)
    return Centerlines(raster=raster, segments=segments_df)


def _build_binary_raster(
    segments_df: pd.DataFrame, template: xr.DataArray
) -> xr.DataArray:
    arr = np.zeros(template.shape, dtype=np.uint8)
    if not segments_df.empty:
        for pixels in segments_df["pixels"]:
            for r, c in pixels:
                if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                    arr[r, c] = 1
    return _wrap_like(arr, template)


def _wrap_like(arr: np.ndarray, template: xr.DataArray) -> xr.DataArray:
    out = xr.DataArray(
        arr, coords=template.coords, dims=template.dims, attrs=template.attrs
    )
    out.rio.write_crs(template.rio.crs, inplace=True)
    out.rio.write_transform(template.rio.transform(), inplace=True)
    return out


__all__ = ["get_centerlines", "Centerlines", "derive_segments", "annotate_segments"]
