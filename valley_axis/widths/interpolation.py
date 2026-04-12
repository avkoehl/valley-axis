import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree


def compute_voronoi(
    centerline_array: np.ndarray,
    mask_array: np.ndarray,
    centerline_widths: np.ndarray,
) -> np.ndarray:
    _, indices = distance_transform_edt(centerline_array == 0, return_indices=True)
    nearest_y, nearest_x = indices[0], indices[1]

    out = np.where(mask_array == 1, centerline_widths[nearest_y, nearest_x], 0.0)
    out = np.where((centerline_array > 0) & (mask_array == 1), centerline_widths, out)
    return out


def compute_idw(
    centerline_array: np.ndarray,
    mask_array: np.ndarray,
    centerline_widths: np.ndarray,
    k: int = 10,
    power: float = 0.75,
) -> np.ndarray:
    cy, cx = np.nonzero(centerline_array > 0)
    k = min(k, len(cy))
    known_widths = centerline_widths[cy, cx]
    tree = cKDTree(np.column_stack((cy, cx)))

    ty, tx = np.nonzero(mask_array == 1)
    distances, indices = tree.query(np.column_stack((ty, tx)), k=k)
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    distances = np.maximum(distances, 1e-12)
    weights = 1.0 / (distances**power)
    interpolated = np.sum(weights * known_widths[indices], axis=1) / np.sum(
        weights, axis=1
    )

    exact_mask = distances[:, 0] <= 1e-12
    interpolated[exact_mask] = known_widths[indices[exact_mask, 0]]

    out = np.full(mask_array.shape, np.nan, dtype=np.float32)
    out[ty, tx] = interpolated
    return out


def compute_laplace(
    centerline_array: np.ndarray,
    mask_array: np.ndarray,
    centerline_widths: np.ndarray,
) -> np.ndarray:
    valid_mask = mask_array == 1
    N = int(np.sum(valid_mask))

    if N == 0:
        return np.zeros_like(mask_array, dtype=np.float64)

    idx_map = np.full(mask_array.shape, -1, dtype=np.int32)
    idx_map[valid_mask] = np.arange(N)

    valid_y, valid_x = np.nonzero(valid_mask)
    pixel_ids = idx_map[valid_y, valid_x]
    is_center = (centerline_array > 0) & valid_mask
    center_mask_1d = is_center[valid_y, valid_x]

    rows, cols, data = [], [], []
    b = np.zeros(N)

    # Dirichlet BCs at centerline pixels
    center_ids = pixel_ids[center_mask_1d]
    rows.append(center_ids)
    cols.append(center_ids)
    data.append(np.ones(len(center_ids)))
    b[center_ids] = centerline_widths[valid_y[center_mask_1d], valid_x[center_mask_1d]]

    # Laplace stencil for non-centerline pixels
    nc_y = valid_y[~center_mask_1d]
    nc_x = valid_x[~center_mask_1d]
    nc_ids = pixel_ids[~center_mask_1d]
    neighbor_counts = np.zeros(len(nc_ids), dtype=np.int32)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = nc_y + dy, nc_x + dx
        in_bounds = (
            (ny >= 0)
            & (ny < mask_array.shape[0])
            & (nx >= 0)
            & (nx < mask_array.shape[1])
        )
        has_neighbor = np.zeros(len(nc_ids), dtype=bool)
        has_neighbor[in_bounds] = valid_mask[ny[in_bounds], nx[in_bounds]]

        rows.append(nc_ids[has_neighbor])
        cols.append(idx_map[ny[has_neighbor], nx[has_neighbor]])
        data.append(np.ones(np.sum(has_neighbor)))
        neighbor_counts += has_neighbor

    rows.append(nc_ids)
    cols.append(nc_ids)
    data.append(-neighbor_counts)

    A = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
    )
    x, info = cg(A, b, x0=centerline_widths[valid_y, valid_x], rtol=1e-4)
    if info != 0:
        print("Warning: Conjugate Gradient solver did not converge")

    out = np.zeros_like(mask_array, dtype=np.float64)
    out[valid_mask] = x
    return out
