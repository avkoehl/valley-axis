import networkx as nx


def calculate_hierarchy(segments):
    """
    1. Rebuilds the pixel graph to strictly identify junctions.
    2. Breaks the pixel graph into 'reaches' (real stream segments ending at tributaries).
    3. Calculates Strahler order based on reach topology.
    4. Groups reaches into contiguous Path labels (longest main stem first).
    """
    if not segments:
        return []

    # 1. Build a universal pixel graph from whatever segments are passed in
    pixel_graph = nx.DiGraph()
    for segment in segments:
        for i in range(len(segment) - 1):
            pixel_graph.add_edge(segment[i], segment[i + 1])

    if len(pixel_graph) == 0:
        return []

    # 2. Extract strict "Reaches" (broken exactly at junctions)
    inflows = [n for n in pixel_graph.nodes() if pixel_graph.in_degree(n) == 0]
    junctions = [n for n in pixel_graph.nodes() if pixel_graph.in_degree(n) > 1]

    # A reach must start at a headwater or a junction
    start_nodes = set(inflows + junctions)

    reaches = {}  # reach_id -> list of pixel coords
    reach_id = 0

    for start in start_nodes:
        if pixel_graph.out_degree(start) == 0:
            continue

        # Walk downstream until the next junction or outlet
        segment = [start]
        current = list(pixel_graph.successors(start))[0]

        while current not in start_nodes and pixel_graph.out_degree(current) > 0:
            segment.append(current)
            current = list(pixel_graph.successors(current))[0]

        segment.append(current)  # Append the closing junction/outlet
        reaches[reach_id] = segment
        reach_id += 1

    # 3. Build a "Reach Graph" to calculate true stream order
    reach_graph = nx.DiGraph()

    # Map start pixels to their reach IDs to connect them
    start_map = {seg[0]: rid for rid, seg in reaches.items()}

    for rid, seg in reaches.items():
        reach_graph.add_node(rid)
        end_node = seg[-1]

        # If this reach ends where another begins, connect them
        if end_node in start_map and start_map[end_node] != rid:
            reach_graph.add_edge(rid, start_map[end_node])

    # 4. Calculate proper Strahler Order on the Reach Graph
    strahler = {}
    try:
        topo_reaches = list(nx.topological_sort(reach_graph))
    except nx.NetworkXUnfeasible:
        topo_reaches = reach_graph.nodes()

    for rid in topo_reaches:
        in_edges = list(reach_graph.predecessors(rid))
        if not in_edges:
            strahler[rid] = 1
        else:
            orders = [strahler[p] for p in in_edges]
            max_order = max(orders)
            # Increment only if two or more tributaries of the SAME max order merge
            if orders.count(max_order) > 1:
                strahler[rid] = max_order + 1
            else:
                strahler[rid] = max_order

    # 5. Label paths by total length (Main stem claiming)
    headwater_reaches = [
        n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0
    ]
    outlet_reaches = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]

    all_paths = []
    for hw in headwater_reaches:
        for out in outlet_reaches:
            try:
                path = nx.shortest_path(reach_graph, source=hw, target=out)
                # Proxy for physical length: total pixels in all reaches in the path
                path_len = sum(len(reaches[r]) for r in path)
                all_paths.append((path_len, path))
            except nx.NetworkXNoPath:
                continue

    # Sort so the longest path (main stem) claims reaches first
    all_paths.sort(key=lambda x: x[0], reverse=True)

    labeled_features = []
    claimed_reaches = set()
    current_path_id = 1

    # 6. Greedily claim reaches to generate final features
    for path_len, path in all_paths:
        claimed_something = False

        for rid in path:
            if rid not in claimed_reaches:
                claimed_reaches.add(rid)
                claimed_something = True

                # Each feature is now exactly one true segment (reach)
                labeled_features.append(
                    {
                        "path_label": current_path_id,
                        "strahler_order": strahler[rid],
                        "nodes": reaches[rid],
                    }
                )
            else:
                # Tributary merges into an already claimed larger stream
                break

        if claimed_something:
            current_path_id += 1

    return labeled_features
