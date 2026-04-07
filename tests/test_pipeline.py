import numpy as np
import pytest
import geopandas as gpd
import xarray as xr
import rioxarray

from valley_axis import derive_axis
from valley_axis.sample_data import get_sample_data


@pytest.fixture(scope="module")
def sample_outputs():
    data = get_sample_data()
    dem = rioxarray.open_rasterio(data["dem"]).squeeze()
    region = rioxarray.open_rasterio(data["region"]).squeeze()
    flowlines = gpd.read_file(data["flowlines"])
    return derive_axis(dem, region, flowlines)


def test_returns_three_outputs(sample_outputs):
    assert len(sample_outputs) == 3


def test_centerlines_gdf(sample_outputs):
    gdf, _, _ = sample_outputs
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0
    assert gdf.geometry.geom_type.isin(["LineString", "MultiLineString"]).all()


def test_centerlines_raster(sample_outputs):
    _, centerlines_raster, _ = sample_outputs
    assert isinstance(centerlines_raster, xr.DataArray)
    assert (centerlines_raster.values > 0).any()


def test_widths_raster(sample_outputs):
    _, _, widths = sample_outputs
    assert isinstance(widths, xr.DataArray) or isinstance(widths, np.ndarray)
    valid = (
        widths[~np.isnan(widths)]
        if isinstance(widths, np.ndarray)
        else widths.values[~np.isnan(widths.values)]
    )
    assert len(valid) > 0
    assert (valid > 0).all(), "All widths inside the mask should be positive"


def test_widths_nan_outside_mask(sample_outputs):
    data = get_sample_data()
    region = rioxarray.open_rasterio(data["region"]).squeeze()
    _, _, widths = sample_outputs

    w = widths if isinstance(widths, np.ndarray) else widths.values
    outside = w[region.values != 1]
    assert np.all(np.isnan(outside)), "Pixels outside the mask should be NaN"
