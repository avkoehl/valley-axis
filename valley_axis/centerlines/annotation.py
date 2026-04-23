import numpy as np
import pandas as pd
import networkx as nx
from collections import deque


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
    path_uid, strahler_order, downstream_segment_id, length, pixels. Segment
    ids are globally unique; path_label resets per network (path_label == 1
    is the mainstem of that network, Strahler-first with upstream-length
    tiebreak). path_uid is a globally unique integer key for each
    (network_id, path_label) pair — use it for rasterization and groupbys.
    """
    records = []
    segment_id_start = 1

    for net_idx, segments in enumerate(segments_by_network, start=1):
        if not segments:
            continue

        net_records = _annotate_one_network(
            segments,
            pixel_size=pixel_size,
            network_id=net_idx,
            segment_id_start=segment_id_start,
        )
        records.extend(net_records)
        segment_id_start += len(net_records)

    segments_df = pd.DataFrame(records)
    if not segments_df.empty:
        segments_df["downstream_segment_id"] = segments_df[
            "downstream_segment_id"
        ].astype("Int64")
        segments_df["path_uid"] = (
            segments_df.groupby(["network_id", "path_label"], sort=True).ngroup() + 1
        )
    return segments_df


# -- per-network annotation -------------------------------------------------


def _annotate_one_network(segments, pixel_size, network_id, segment_id_start):
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
    local_path = _claim_paths(reach_graph, segments, strahler, pixel_size)

    records = []
    for i, seg in enumerate(segments):
        ds_local = downstream_local.get(i)
        ds_global = segment_id_start + ds_local if ds_local is not None else pd.NA
        records.append(
            {
                "segment_id": segment_id_start + i,
                "network_id": network_id,
                "path_label": local_path[i],
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


def _claim_paths(reach_graph, segments, strahler, pixel_size):
    """Label paths within a single network.

    Walks upstream from each outlet, picking the highest-Strahler predecessor
    at each junction (upstream-length tiebreak). Non-main branches become
    subsequent paths. Ordering is deterministic: longer branches get lower
    path_ids, so path_label == 1 is the mainstem of this network.
    """
    # upstream length per reach: longest headwater->this, including own length
    upstream_length = {}
    for rid in nx.topological_sort(reach_graph):
        own = _segment_length(segments[rid], pixel_size)
        preds = list(reach_graph.predecessors(rid))
        upstream_length[rid] = own + (
            max(upstream_length[p] for p in preds) if preds else 0.0
        )

    path_id = {}
    next_id = 1
    outlets = [n for n in reach_graph.nodes() if reach_graph.out_degree(n) == 0]
    # process longest outlet first so its mainstem gets id 1
    outlets.sort(key=lambda o: upstream_length.get(o, 0), reverse=True)
    queue = deque(outlets)

    while queue:
        start = queue.popleft()
        if start in path_id:
            continue
        current = start
        while True:
            path_id[current] = next_id
            preds = [p for p in reach_graph.predecessors(current) if p not in path_id]
            if not preds:
                break
            max_s = max(strahler[p] for p in preds)
            candidates = [p for p in preds if strahler[p] == max_s]
            if len(candidates) == 1:
                main = candidates[0]
            else:
                max_len = max(upstream_length[p] for p in candidates)
                main = next(p for p in candidates if upstream_length[p] == max_len)
            # non-main branches become subsequent paths; longer first
            siblings = sorted(
                [p for p in preds if p != main],
                key=lambda x: upstream_length[x],
                reverse=True,
            )
            for p in siblings:
                queue.append(p)
            current = main
        next_id += 1

    return path_id


def _segment_length(pixels, pixel_size):
    n_card = n_diag = 0
    for (r1, c1), (r2, c2) in zip(pixels[:-1], pixels[1:]):
        if abs(r1 - r2) + abs(c1 - c2) == 2:
            n_diag += 1
        else:
            n_card += 1
    return pixel_size * (n_card + np.sqrt(2) * n_diag)
