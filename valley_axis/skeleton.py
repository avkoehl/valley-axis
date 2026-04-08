import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_holes
from skimage.graph import MCP_Geometric


def raster_skeletonization(mask_array, hole_size_threshold=500):
    """
    Fills small holes in the mask to prevent skeleton loops,
    then computes the morphological skeleton.
    """
    bool_mask = mask_array == 1

    # Fill holes smaller than the threshold (in pixels)
    filled_mask = remove_small_holes(bool_mask, max_size=hole_size_threshold)

    # Compute 1-pixel wide medial axis
    skeleton = skeletonize(filled_mask)

    return np.where(skeleton, 1, 0).astype(np.uint8)


def raster_skeleton_to_graph(skeleton_array):
    """
    Converts the 1-pixel wide skeleton array into an 8-connected NetworkX graph.
    """
    G = nx.Graph()
    rows, cols = np.nonzero(skeleton_array)
    nodes = set(zip(rows, cols))

    for r, c in nodes:
        G.add_node((r, c))
        # Check 8-connected neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = r + dr, c + dc
                if (nr, nc) in nodes:
                    weight = np.sqrt(dr**2 + dc**2)
                    G.add_edge((r, c), (nr, nc), weight=weight)

    return G


def filter_inlets(inlets, mask_array, pixel_size, distance_threshold=100):
    """
    Drops inlets that are deep in the valley floor (distance to edge > threshold).
    """
    # Distance to the nearest background (0) pixel
    dist_array = distance_transform_edt(mask_array == 1) * pixel_size

    filtered_inlets = []
    for r, c in inlets:
        if dist_array[r, c] <= distance_threshold:
            filtered_inlets.append((r, c))

    return filtered_inlets


def mask_constrained_snap(inlets, outlet, skeleton_array, mask_array):
    """
    Finds the shortest valid path from inlets/outlets to the skeleton.
    """
    # Using np.inf is perfectly fine as long as endpoints are inside the mask
    penalty = np.where(mask_array == 1, 1.0, np.inf)
    mcp = MCP_Geometric(penalty)

    # Cast explicitly to native ints
    rows, cols = np.nonzero(skeleton_array)
    skeleton_pixels = [(int(r), int(c)) for r, c in zip(rows, cols)]

    if not skeleton_pixels:
        return {}, None

    mcp.find_costs(starts=skeleton_pixels)

    snapped_inlets = {}
    for r, c in inlets:
        # Traceback returns [skeleton_pixel, ..., inlet_pixel]
        path = mcp.traceback([int(r), int(c)])
        if path:
            clean_path = [(int(p[0]), int(p[1])) for p in path]
            snapped_inlets[(int(r), int(c))] = {
                "snap_point": clean_path[0],  # path[0] is the skeleton intersection
                "path": clean_path[
                    ::-1
                ],  # Reverse it so it flows Downstream: inlet -> skeleton
            }

    # Snap outlet
    out_r, out_c = outlet
    out_path = mcp.traceback([int(out_r), int(out_c)])
    snapped_outlet = None

    if out_path:
        # out_path returns [skeleton_pixel, ..., outlet_pixel]
        clean_out_path = [(int(p[0]), int(p[1])) for p in out_path]
        snapped_outlet = {
            "snap_point": clean_out_path[0],  # path[0] is the skeleton intersection
            "path": clean_out_path,  # Keep as-is so it flows Downstream: skeleton -> outlet
        }

    return snapped_inlets, snapped_outlet


def route_paths(graph, snapped_inlets, snapped_outlet):
    """
    Routes from snapped inlets to snapped outlet via the skeleton graph,
    then deduplicates overlapping segments into a clean directed tree.
    """
    if not snapped_outlet:
        return []

    raw_paths = []
    out_node = snapped_outlet["snap_point"]

    # 1. Add the path connecting the skeleton to the true outlet
    raw_paths.append(snapped_outlet["path"])

    # 2. Add inlet connection paths & the main skeleton shortest paths
    for inlet, data in snapped_inlets.items():
        in_node = data["snap_point"]
        raw_paths.append(data["path"])

        try:
            skel_path = nx.shortest_path(
                graph, source=in_node, target=out_node, weight="weight"
            )
            raw_paths.append(skel_path)
        except nx.NetworkXNoPath:
            continue

    # 3. Deduplicate overlapping paths into a directed graph (Reused logic)
    G = nx.DiGraph()
    for path in raw_paths:
        if len(path) < 2:
            continue
        for i in range(len(path) - 1):
            G.add_edge(tuple(path[i]), tuple(path[i + 1]))

    # 4. Identify junctions and inflows
    for node in G.nodes():
        if G.in_degree(node) == 0 and G.out_degree(node) > 0:
            G.nodes[node]["inflow"] = True
        if G.in_degree(node) > 1 and G.out_degree(node) > 0:
            G.nodes[node]["junction"] = True

    # 5. Extract clean segments
    start_nodes = [
        n for n, d in G.nodes(data=True) if d.get("inflow") or d.get("junction")
    ]

    segments = []
    for start in start_nodes:
        if G.out_degree(start) == 0:
            continue

        segment = [start]
        current = list(G.successors(start))[0]

        while (current not in start_nodes) and (G.out_degree(current) > 0):
            segment.append(current)
            current = list(G.successors(current))[0]

        segment.append(current)
        segments.append(segment)

    return segments
