import warnings

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from .centerlines import Centerlines


def get_widths(
    centerlines: Centerlines,
    mask: xr.DataArray,
    allocation: xr.DataArray | None = None,
    method: str = "laplace",
) -> xr.DataArray:
    """
    Compute valley width at every pixel of the mask.
    Exact widths are taken at centerline pixels (2 * distance-to-edge) and
    interpolated across the mask. When `allocation` is provided (a
    path-labeled raster), each path's territory is interpolated independently
    — preserving continuity along paths while respecting boundaries between
    paths.
    method: "laplace" (smooth diffusion) or "nearest" (Voronoi NN).
    """
    if method not in ("laplace", "nearest"):
        raise ValueError(f"method must be 'laplace' or 'nearest', got '{method}'")
    interp = _laplace if method == "laplace" else _nearest
    pixel_size = float(abs(mask.rio.resolution()[0]))
    mask_array = mask.values == 1
    path_raster = centerlines.label_by_path()
    path_centerline_array = path_raster.values
    radius_pixels = distance_transform_edt(mask_array)
    centerline_widths = np.where(
        path_centerline_array > 0, radius_pixels * pixel_size * 2, 0.0
    )
    out = np.full(mask_array.shape, np.nan, dtype=np.float64)
    if allocation is None:
        result = interp(path_centerline_array, mask_array, centerline_widths)
        out[mask_array] = result[mask_array]
    else:
        alloc = allocation.values
        for path_id in np.unique(alloc[alloc > 0]):
            region_mask = (alloc == path_id) & mask_array
            region_centerlines = np.where(
                path_centerline_array == path_id, path_centerline_array, 0
            )
            if not np.any(region_centerlines > 0) or not np.any(region_mask):
                continue
            result = interp(region_centerlines, region_mask, centerline_widths)
            out[region_mask] = result[region_mask]
    widths = xr.DataArray(out, coords=mask.coords, dims=mask.dims, attrs=mask.attrs)
    widths = widths.where(mask_array)
    widths.rio.write_crs(mask.rio.crs, inplace=True)
    widths.rio.write_transform(mask.rio.transform(), inplace=True)
    widths.rio.write_nodata(np.nan, inplace=True)
    return widths


def _nearest(centerline_array, mask_array, centerline_widths):
    _, indices = distance_transform_edt(centerline_array == 0, return_indices=True)
    nearest_y, nearest_x = indices[0], indices[1]
    out = np.where(mask_array, centerline_widths[nearest_y, nearest_x], 0.0)
    out = np.where((centerline_array > 0) & mask_array, centerline_widths, out)
    return out


def _laplace(centerline_array, mask_array, centerline_widths):
    N = int(np.sum(mask_array))
    if N == 0:
        return np.zeros_like(mask_array, dtype=np.float64)

    idx_map = np.full(mask_array.shape, -1, dtype=np.int32)
    idx_map[mask_array] = np.arange(N)

    valid_y, valid_x = np.nonzero(mask_array)
    pixel_ids = idx_map[valid_y, valid_x]
    is_center = (centerline_array > 0) & mask_array
    center_1d = is_center[valid_y, valid_x]

    rows, cols, data = [], [], []
    b = np.zeros(N)

    # Dirichlet BCs at centerline pixels
    center_ids = pixel_ids[center_1d]
    rows.append(center_ids)
    cols.append(center_ids)
    data.append(np.ones(len(center_ids)))
    b[center_ids] = centerline_widths[valid_y[center_1d], valid_x[center_1d]]

    # 8-connected Laplace stencil for non-centerline pixels
    # (weights: 1 for cardinal, 1/√2 for diagonal — preserves isotropy and
    # prevents isolation of pixels only connected diagonally within the mask)
    nc_y = valid_y[~center_1d]
    nc_x = valid_x[~center_1d]
    nc_ids = pixel_ids[~center_1d]
    neighbor_weight_sum = np.zeros(len(nc_ids), dtype=np.float64)

    inv_sqrt2 = 1.0 / np.sqrt(2)
    offsets = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, inv_sqrt2),
        (-1, 1, inv_sqrt2),
        (1, -1, inv_sqrt2),
        (1, 1, inv_sqrt2),
    ]
    for dy, dx, w in offsets:
        ny, nx = nc_y + dy, nc_x + dx
        in_bounds = (
            (ny >= 0)
            & (ny < mask_array.shape[0])
            & (nx >= 0)
            & (nx < mask_array.shape[1])
        )
        has_neighbor = np.zeros(len(nc_ids), dtype=bool)
        has_neighbor[in_bounds] = mask_array[ny[in_bounds], nx[in_bounds]]

        rows.append(nc_ids[has_neighbor])
        cols.append(idx_map[ny[has_neighbor], nx[has_neighbor]])
        data.append(np.full(int(np.sum(has_neighbor)), w))
        neighbor_weight_sum[has_neighbor] += w

    rows.append(nc_ids)
    cols.append(nc_ids)
    data.append(-neighbor_weight_sum)

    A = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
    )
    x, info = cg(A, b, x0=centerline_widths[valid_y, valid_x], rtol=1e-4)
    if info != 0:
        warnings.warn("Conjugate gradient solver did not converge")

    out = np.zeros_like(mask_array, dtype=np.float64)
    out[mask_array] = x
    return out
