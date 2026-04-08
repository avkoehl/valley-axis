import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.ndimage import distance_transform_edt


# inspired by: difference is that we use the exact widths as Dirichlet boundary
# conditions at the centerline, and solve for the rest of the valley using
# Laplace's equation. This allows us to get a smooth interpolation of widths
# across the entire valley, rather than just at the centerline.
# https://onlinelibrary.wiley.com/doi/10.1002/1097-0193(200009)11:1%3C12::AID-HBM20%3E3.0.CO;2-K
# Three-dimensional mapping of cortical thickness using Laplace's Equation, Jones 2000
def compute_laplace(centerline_array, mask_array, pixel_size):
    """
    Computes continuous valley widths using Harmonic Interpolation (Laplace Equation).
    """
    # 1. Compute exact widths everywhere (2 * distance to nearest boundary)
    radius_pixels = distance_transform_edt(mask_array == 1)
    exact_widths = radius_pixels * pixel_size * 2

    valid_mask = mask_array == 1
    N = np.sum(valid_mask)

    if N == 0:
        return np.zeros_like(mask_array, dtype=np.float64)

    # 2. Map 2D coordinates to a 1D linear system
    idx_map = np.full(mask_array.shape, -1, dtype=np.int32)
    idx_map[valid_mask] = np.arange(N)

    is_center = (centerline_array > 0) & valid_mask
    valid_y, valid_x = np.nonzero(valid_mask)
    pixel_ids = idx_map[valid_y, valid_x]

    center_mask_1d = is_center[valid_y, valid_x]

    # Prepare sparse matrix arrays
    rows = []
    cols = []
    data = []
    b = np.zeros(N)

    # 3. Apply Dirichlet boundary conditions (Centerlines)
    center_ids = pixel_ids[center_mask_1d]
    rows.append(center_ids)
    cols.append(center_ids)
    data.append(np.ones(len(center_ids)))
    b[center_ids] = exact_widths[valid_y[center_mask_1d], valid_x[center_mask_1d]]

    # 4. Apply Laplace stencil (Neumann boundaries at walls)
    non_center_mask = ~center_mask_1d
    nc_y = valid_y[non_center_mask]
    nc_x = valid_x[non_center_mask]
    nc_ids = pixel_ids[non_center_mask]

    neighbor_counts = np.zeros(len(nc_ids), dtype=np.int32)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dy, dx in directions:
        ny, nx = nc_y + dy, nc_x + dx

        # Check bounds
        in_bounds = (
            (ny >= 0)
            & (ny < mask_array.shape[0])
            & (nx >= 0)
            & (nx < mask_array.shape[1])
        )

        # Check if neighbor is within the valley mask
        has_neighbor = np.zeros(len(nc_ids), dtype=bool)
        has_neighbor[in_bounds] = valid_mask[ny[in_bounds], nx[in_bounds]]

        # Add edges for valid neighbors (+1 off-diagonal)
        n_ids = idx_map[ny[has_neighbor], nx[has_neighbor]]
        rows.append(nc_ids[has_neighbor])
        cols.append(n_ids)
        data.append(np.ones(np.sum(has_neighbor)))

        neighbor_counts += has_neighbor

    # Subtract neighbor count from main diagonal (-N_neighbors)
    rows.append(nc_ids)
    cols.append(nc_ids)
    data.append(-neighbor_counts)

    # 5. Build sparse matrix and solve A * x = b
    A = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
    )
    x0 = exact_widths[valid_y, valid_x]  # Initial guess
    x, info = cg(A, b, x0=x0, rtol=1e-4)
    if info != 0:
        print("Warning: Conjugate Gradient solver did not converge")

    # 6. Map back to 2D raster
    final_widths = np.zeros_like(mask_array, dtype=np.float64)
    final_widths[valid_mask] = x

    return final_widths
