# valley-axis

Valley floor centerline extraction and continuous width computation for river corridors.

Implements a least-cost path algorithm (adapted from the glacial centerlines
method) to route valley axis centerlines through a valley floor mask, guided by
a DEM and stream network. Valley widths are then computed via inverse distance
weighting (IDW) from the centerline outward.

## Installation

```bash
pip install git+https://github.com/avkoehl/valley-axis.git
```

For development:

```bash
git clone https://github.com/avkoehl/valley-axis.git
cd valley-axis
uv sync --extra dev
```

## Inputs

| Input | Type | Description |
|---|---|---|
| `dem` | `xr.DataArray` | Digital elevation model with spatial metadata |
| `region_raster` | `xr.DataArray` | Valley floor mask (1 = valid, 0 = outside) |
| `flowlines` | `gpd.GeoDataFrame` | Stream network linestrings |

## Usage

```python
import rioxarray
import geopandas as gpd
from valley_axis import derive_axis

dem = rioxarray.open_rasterio("dem.tif").squeeze()
region = rioxarray.open_rasterio("region.tif").squeeze()
flowlines = gpd.read_file("flowlines.gpkg")

centerlines_gdf, centerlines_raster, widths_raster = derive_axis(dem, region, flowlines)
```

### Outputs

- **`centerlines_gdf`** — `GeoDataFrame` of routed centerline segments as LineStrings
- **`centerlines_raster`** — `xr.DataArray` raster of centerline segment IDs
- **`widths_raster`** — `np.ndarray` of continuous valley widths (metres), NaN outside the mask

### Cost function parameters

The routing cost surface can be tuned via keyword arguments:

```python
centerlines_gdf, centerlines_raster, widths_raster = derive_axis(
    dem, region, flowlines,
    f1=1000, a=4.25,   # distance penalty: scale and exponent
    f2=3000, b=3.5,    # elevation penalty: scale and exponent
)
```

## Sample data

```python
import rioxarray
import geopandas as gpd

data = get_sample_data()
dem = rioxarray.open_rasterio(data["dem"]).squeeze()
region = rioxarray.open_rasterio(data["region"]).squeeze()
flowlines = gpd.read_file(data["flowlines"])
```

See `examples/valley_axis_demo.ipynb` for a full walkthrough.

## Reference

Kienholz, C., Rich, J. L., Arendt, A. A., & Hock, R. (2014). A new method for deriving glacier centerlines applied to glaciers in Alaska and northwest Canada. *The Cryosphere*, 8(2), 503–519. https://doi.org/10.5194/tc-8-503-2014
