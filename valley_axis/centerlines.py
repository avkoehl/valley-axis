import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric


def build_cost_graph(
    dem_array, mask_array, pixel_size, f1=1000, a=4.25, f2=3000, b=3.5
):
    """
    Computes the penalty surface and initializes the MCP_Geometric routing object.
    """
    # 1. Distance penalty
    distance = distance_transform_edt(mask_array) * pixel_size
    d_max = np.nanmax(distance)

    if d_max == 0:
        dist_penalty = np.zeros_like(distance)
    else:
        dist_penalty = (d_max - distance) / d_max * f1
        dist_penalty = np.power(dist_penalty, a)

    # 2. Elevation penalty
    z_min = np.nanmin(dem_array)
    z_max = np.nanmax(dem_array)

    if z_max == z_min:
        elev_penalty = np.zeros_like(dem_array)
    else:
        elev_penalty = (dem_array - z_min) / (z_max - z_min) * f2
        elev_penalty = np.power(elev_penalty, b)

    # 3. Combine and mask
    penalty = dist_penalty + elev_penalty
    penalty = np.where(mask_array == 1, penalty, np.inf)

    # Return the initialized graph object
    return MCP_Geometric(penalty)


def route_segments(mcp, inlets, outlet):
    """
    Routes paths from inlets to the outlet and extracts non-overlapping segments.
    """
    out_row, out_col = outlet

    # 1. Trace raw paths
    paths = []
    for row, col in inlets:
        mcp.find_costs(starts=[[row, col]], ends=[[out_row, out_col]])
        path = mcp.traceback([out_row, out_col])
        paths.append(path)

    # 2. Build directed graph to remove overlaps
    G = nx.DiGraph()
    for path in paths:
        for i in range(len(path) - 1):
            G.add_edge(tuple(path[i]), tuple(path[i + 1]))

    # 3. Identify junctions and inflows
    for node in G.nodes():
        if G.in_degree(node) == 0 and G.out_degree(node) > 0:
            G.nodes[node]["inflow"] = True
        if G.in_degree(node) > 1 and G.out_degree(node) > 0:
            G.nodes[node]["junction"] = True

    # 4. Extract clean segments
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
