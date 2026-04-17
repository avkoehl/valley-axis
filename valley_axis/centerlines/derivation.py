import numpy as np
import networkx as nx
import xarray as xr
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.graph import MCP_Geometric


Segment = list[tuple[int, int]]
Network = tuple[list[tuple[int, int]], tuple[int, int]]


def derive_segments(
    mask: xr.DataArray,
    networks: list[Network],
    inlet_distance_threshold: float = 100.0,
) -> list[list[Segment]]:
    """
    Derive centerline segments per network.

    Returns a list (one element per input network) of segment lists.
    Each segment is an ordered list of (row, col) pixel tuples, upstream→downstream.
    Networks that yield no valid segments produce an empty list.
    """
    mask_array = mask.values == 1
    pixel_size = float(abs(mask.rio.resolution()[0]))

    skeleton = skeletonize(mask_array).astype(np.uint8)
    graph = _skeleton_to_graph(skeleton)

    out = []
    for inlets, outlet in networks:
        inlets = _filter_inlets(
            inlets, mask_array, pixel_size, inlet_distance_threshold
        )
        if not inlets:
            out.append([])
            continue

        snapped_inlets, snapped_outlet = _snap_endpoints(
            inlets, outlet, skeleton, mask_array
        )
        if snapped_outlet is None:
            out.append([])
            continue

        raw_paths = _route_raw_paths(graph, snapped_inlets, snapped_outlet)
        out.append(_extract_segments(raw_paths))

    return out


# -- skeleton graph ---------------------------------------------------------


def _skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    rows, cols = np.nonzero(skeleton)
    nodes = set(zip(rows.tolist(), cols.tolist()))

    for r, c in nodes:
        G.add_node((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in nodes:
                    G.add_edge((r, c), (r + dr, c + dc), weight=np.hypot(dr, dc))
    return G


# -- endpoint prep ----------------------------------------------------------


def _filter_inlets(inlets, mask_array, pixel_size, threshold):
    dist = distance_transform_edt(mask_array) * pixel_size
    return [(r, c) for r, c in inlets if dist[r, c] <= threshold]


def _snap_endpoints(inlets, outlet, skeleton, mask_array):
    """Snap inlets and outlet to the skeleton via least-cost paths within the mask."""
    penalty = np.where(mask_array, 1.0, np.inf)
    mcp = MCP_Geometric(penalty)

    rows, cols = np.nonzero(skeleton)
    skeleton_pixels = [(int(r), int(c)) for r, c in zip(rows, cols)]
    if not skeleton_pixels:
        return {}, None

    mcp.find_costs(starts=skeleton_pixels)

    snapped_inlets = {}
    for r, c in inlets:
        path = mcp.traceback([int(r), int(c)])
        if not path:
            continue
        clean = [(int(p[0]), int(p[1])) for p in path]
        snapped_inlets[(r, c)] = {"snap_point": clean[0], "path": clean[::-1]}

    out_path = mcp.traceback([int(outlet[0]), int(outlet[1])])
    snapped_outlet = None
    if out_path:
        clean = [(int(p[0]), int(p[1])) for p in out_path]
        snapped_outlet = {"snap_point": clean[0], "path": clean}

    return snapped_inlets, snapped_outlet


# -- routing ----------------------------------------------------------------


def _route_raw_paths(graph, snapped_inlets, snapped_outlet):
    raw_paths = [snapped_outlet["path"]]
    out_node = snapped_outlet["snap_point"]

    for data in snapped_inlets.values():
        raw_paths.append(data["path"])
        try:
            raw_paths.append(
                nx.shortest_path(
                    graph, source=data["snap_point"], target=out_node, weight="weight"
                )
            )
        except nx.NetworkXNoPath:
            continue
    return raw_paths


def _extract_segments(raw_paths) -> list[Segment]:
    """Deduplicate overlapping raw paths into a directed tree, then break at junctions."""
    G = nx.DiGraph()
    for path in raw_paths:
        for a, b in zip(path[:-1], path[1:]):
            G.add_edge(tuple(a), tuple(b))

    if len(G) == 0:
        return []

    inflows = {n for n in G.nodes() if G.in_degree(n) == 0}
    junctions = {n for n in G.nodes() if G.in_degree(n) > 1}
    breakpoints = inflows | junctions

    segments = []
    for start in breakpoints:
        if G.out_degree(start) == 0:
            continue
        seg = [start]
        current = next(iter(G.successors(start)))
        while current not in breakpoints and G.out_degree(current) > 0:
            seg.append(current)
            current = next(iter(G.successors(current)))
        seg.append(current)
        segments.append(seg)
    return segments
