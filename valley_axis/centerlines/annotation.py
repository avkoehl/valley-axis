import numpy as np
import pandas as pd
import networkx as nx


Segment = list[tuple[int, int]]


def annotate_segments(
    segments_by_network: list[list[Segment]],
    pixel_size: float,
) -> pd.DataFrame:
    """
    Annotate centerline segments with topology.

    Input: list (one per network) of segment lists. Segments are ordered
    (row, col) pixel tuples, upstream→downstream.

    Returns a DataFrame with columns: segment_id, network_id, path_label,
    strahler_order, downstream_segment_id, length, pixels. Segment ids and
    path labels are globally unique across networks.
    """
    records = []
    segment_id_start = 1
    path_label_offset = 0

    for net_idx, segments in enumerate(segments_by_network, start=1):
        if not segments:
            continue

        net_records = _annotate_one_network(
            segments,
            pixel_size=pixel_size,
            network_id=net_idx,
            segment_id_start=segment_id_start,
            path_label_offset=path_label_offset,
        )
        records.extend(net_records)
        segment_id_start += len(net_records)
        path_label_offset = max(r["path_label"] for r in net_records)

    segments_df = pd.DataFrame(records)
    if not segments_df.empty:
        segments_df["downstream_segment_id"] = segments_df[
            "downstream_segment_id"
        ].astype("Int64")
    return segments_df


# -- per-network annotation -------------------------------------------------


def _annotate_one_network(
    segments, pixel_size, network_id, segment_id_start, path_label_offset
):
    start_map = {seg[0]: i for i, seg in enumerate(segments)}
    reach_graph = nx.DiGraph()
    reach_graph.add_nodes_from(range(len(segments)))
    downstream_local = {}
    for i, seg in enumerate(segments):
        end = seg[-1]
        if end in start_map and start_map[end] != i:
            reach_graph.add_edge(i, start_map[end])
            downstream_local[i] = start_map[end]

    strahler = _strahler(reach_graph)
    local_path = _claim_paths(reach_graph, segments)

    records = []
    for i, seg in enumerate(segments):
        ds_local = downstream_local.get(i)
        ds_global = segment_id_start + ds_local if ds_local is not None else pd.NA
        records.append(
            {
                "segment_id": segment_id_start + i,
                "network_id": network_id,
                "path_label": path_label_offset + local_path[i],
                "strahler_order": strahler[i],
                "downstream_segment_id": ds_global,
                "length": _segment_length(seg, pixel_size),
                "pixels": seg,
            }
        )
    return records


def _strahler(reach_graph):
    try:
        order = list(nx.topological_sort(reach_graph))
    except nx.NetworkXUnfeasible:
        order = list(reach_graph.nodes())

    strahler = {}
    for rid in order:
        preds = list(reach_graph.predecessors(rid))
        if not preds:
            strahler[rid] = 1
        else:
            orders = [strahler[p] for p in preds]
            m = max(orders)
            strahler[rid] = m + 1 if orders.count(m) > 1 else m
    return strahler


def _claim_paths(reach_graph, segments):
    """Greedy main-stem path labeling: longest headwater→outlet path claims first."""
    headwaters = [n for n in reach_graph.nodes() if reach_graph.in_degree(n) == 0]
    outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]

    candidates = []
    for hw in headwaters:
        for out in outlets:
            try:
                path = nx.shortest_path(reach_graph, source=hw, target=out)
                plen = sum(len(segments[r]) for r in path)
                candidates.append((plen, path))
            except nx.NetworkXNoPath:
                continue
    candidates.sort(key=lambda x: x[0], reverse=True)

    reach_to_path = {}
    current = 1
    for _, path in candidates:
        claimed_any = False
        for rid in path:
            if rid in reach_to_path:
                break
            reach_to_path[rid] = current
            claimed_any = True
        if claimed_any:
            current += 1
    return reach_to_path


def _segment_length(pixels, pixel_size):
    n_card = n_diag = 0
    for (r1, c1), (r2, c2) in zip(pixels[:-1], pixels[1:]):
        if abs(r1 - r2) + abs(c1 - c2) == 2:
            n_diag += 1
        else:
            n_card += 1
    return pixel_size * (n_card + np.sqrt(2) * n_diag)
